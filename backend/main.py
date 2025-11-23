import os
# 强制禁用 GPU，避免环境配置问题导致 DeepAR 失败
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import shutil
import sqlite3
import pandas as pd
import io
import zipfile
import sys
import numpy as np
import math
from typing import List, Optional, Dict, Any
import uuid
from pydantic import BaseModel
from datetime import datetime
import json
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess

app = FastAPI(title="GeneralPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data", "db")
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts", "user_crawlers")
METADATA_DB = os.path.join(BASE_DIR, "data", "metadata.db")
RESULTS_DB = os.path.join(BASE_DIR, "data", "results.db")

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

# --- 初始化调度器 ---
scheduler = BackgroundScheduler()
scheduler.start()

# --- 初始化数据库 ---
def init_dbs():
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            name TEXT,
            db_file TEXT,
            table_name TEXT,
            config TEXT,
            status TEXT,
            created_at TEXT,
            error_msg TEXT,
            cron_expression TEXT,
            trigger_crawler TEXT
        )
    ''')
    
    # 爬虫表
    c.execute('''
        CREATE TABLE IF NOT EXISTS crawlers (
            id TEXT PRIMARY KEY,
            name TEXT,
            filename TEXT,
            cron_expression TEXT,
            last_run TEXT,
            status TEXT
        )
    ''')
    
    # 爬虫日志表
    c.execute('''
        CREATE TABLE IF NOT EXISTS crawler_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crawler_id TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            log_content TEXT
        )
    ''')

    # 预测任务日志表
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            trigger_type TEXT,
            log_content TEXT
        )
    ''')
    
    # 尝试为旧数据库添加新列 (简单迁移)
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN cron_expression TEXT")
    except: pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN trigger_crawler TEXT")
    except: pass
    
    conn.commit()
    conn.close()

init_dbs()

# --- Models ---
class FileInfo(BaseModel):
    name: str
    size: int
    path: str

class TaskConfig(BaseModel):
    taskName: str
    dbFile: str
    table: str
    mapping: Dict[str, Any]
    models: List[str]
    params: Dict[str, Any]
    cron: Optional[str] = None
    trigger_crawler: Optional[str] = None

class Task(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    error_msg: Optional[str] = None
    cron_expression: Optional[str] = None
    trigger_crawler: Optional[str] = None

class Crawler(BaseModel):
    id: str
    name: str
    filename: str
    cron_expression: Optional[str] = None
    status: str

class CrawlerLog(BaseModel):
    id: int
    start_time: str
    end_time: Optional[str]
    status: str
    log_content: Optional[str]

class PredictionLog(BaseModel):
    id: int
    task_id: str
    start_time: str
    end_time: Optional[str]
    status: str
    trigger_type: Optional[str]
    log_content: Optional[str]

# --- Log Manager ---
class LogManager:
    def __init__(self):
        # Key: log_key (e.g., "crawler:uuid" or "task:uuid"), Value: list of log lines
        self.active_logs: Dict[str, List[str]] = {}
        
    def append(self, log_key: str, message: str):
        if log_key not in self.active_logs:
            self.active_logs[log_key] = []
        self.active_logs[log_key].append(message)
        
    def get_logs(self, log_key: str) -> str:
        return "\n".join(self.active_logs.get(log_key, []))
        
    def clear(self, log_key: str):
        if log_key in self.active_logs:
            del self.active_logs[log_key]

log_manager = LogManager()

@app.get("/api/logs/live")
def get_live_logs(type: str, id: str):
    log_key = f"{type}:{id}"
    content = log_manager.get_logs(log_key)
    return {"content": content}

# --- 文件管理 API ---
@app.get("/api/files/db", response_model=List[FileInfo])
def list_db_files():
    files = []
    if not os.path.exists(DB_DIR): return []
    for f in os.listdir(DB_DIR):
        if f.endswith('.db'):
            path = os.path.join(DB_DIR, f)
            files.append({
                "name": f,
                "size": os.path.getsize(path),
                "path": path
            })
    return files

@app.post("/api/files/upload")
async def upload_db_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.db'):
        raise HTTPException(status_code=400, detail="Only .db files are allowed")
    file_path = os.path.join(DB_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.delete("/api/files/db/{filename}")
def delete_db_file(filename: str):
    file_path = os.path.join(DB_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/db/{filename}/tables")
def list_tables(filename: str):
    db_path = os.path.join(DB_DIR, filename)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/db/{filename}/preview/{table}")
def preview_table(filename: str, table: str, limit: int = 20):
    db_path = os.path.join(DB_DIR, filename)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", conn)
        conn.close()
        df = df.where(pd.notnull(df), None)
        return {
            "columns": list(df.columns),
            "data": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 爬虫 API (升级版) ---

@app.post("/api/crawler/upload")
async def upload_crawler_script(file: UploadFile = File(...), cron: Optional[str] = None):
    if not file.filename.endswith('.py'):
        raise HTTPException(status_code=400, detail="Only .py files are allowed")
    
    file_path = os.path.join(SCRIPT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 入库
    crawler_id = str(uuid.uuid4())
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("SELECT id FROM crawlers WHERE filename = ?", (file.filename,))
    existing = c.fetchone()
    
    if existing:
        crawler_id = existing[0]
        c.execute("UPDATE crawlers SET cron_expression = ?, status = 'idle' WHERE id = ?", (cron, crawler_id))
    else:
        c.execute("INSERT INTO crawlers (id, name, filename, cron_expression, status) VALUES (?, ?, ?, ?, 'idle')",
                  (crawler_id, file.filename, file.filename, cron))
    conn.commit()
    conn.close()
    
    if cron:
        schedule_crawler(crawler_id, file.filename, cron)
        
    return {"filename": file.filename, "status": "uploaded", "id": crawler_id}

@app.get("/api/crawler/list", response_model=List[Crawler])
def list_crawlers():
    # 从数据库读取，而不是文件系统，因为有 cron 信息
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM crawlers")
    rows = c.fetchall()
    conn.close()
    
    # 同步文件系统：如果有新文件没入库，或者库里有但文件没了（虽然不推荐，但为了健壮性）
    # 这里简化：只返回库里的
    return [dict(row) for row in rows]

@app.get("/api/crawler/{crawler_id}/logs", response_model=List[CrawlerLog])
def get_crawler_logs(crawler_id: str):
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM crawler_logs WHERE crawler_id = ? ORDER BY start_time DESC LIMIT 50", (crawler_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def schedule_crawler(crawler_id, filename, cron_str):
    if scheduler.get_job(crawler_id):
        scheduler.remove_job(crawler_id)
    try:
        scheduler.add_job(
            run_crawler_job,
            CronTrigger.from_crontab(cron_str),
            id=crawler_id,
            args=[crawler_id, filename]
        )
        print(f"Scheduled crawler {filename} with cron: {cron_str}")
    except Exception as e:
        print(f"Failed to schedule crawler: {e}")

def run_crawler_job(crawler_id, filename):
    print(f"Running crawler job: {filename}")
    script_path = os.path.join(SCRIPT_DIR, filename)
    start_time = datetime.now().isoformat()
    log_key = f"crawler:{crawler_id}"
    log_manager.clear(log_key)
    
    # 记录开始
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("INSERT INTO crawler_logs (crawler_id, start_time, status) VALUES (?, ?, 'running')", (crawler_id, start_time))
    log_id = c.lastrowid
    conn.commit()
    conn.close()
    
    status = "failed"
    final_log_content = ""
    
    try:
        # 使用 Popen 实现实时日志捕获
        # env={**os.environ, "PYTHONUNBUFFERED": "1"} 确保 Python 输出不缓冲
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 合并 stderr 到 stdout
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # 解决 Windows 下 print 特殊字符 (如 \ufffd) 可能导致的 GBK 编码错误
                # 方法：先 encode 成控制台编码(或 gbk)并替换错误，再 decode 回来
                clean_line = line.rstrip()
                try:
                    print(f"[Crawler {filename}] {clean_line}")
                except UnicodeEncodeError:
                    # 如果打印失败，尝试安全打印
                    safe_line = clean_line.encode(sys.stdout.encoding or 'gbk', errors='replace').decode(sys.stdout.encoding or 'gbk')
                    print(f"[Crawler {filename}] {safe_line}")
                
                log_manager.append(log_key, clean_line)
        
        rc = process.poll()
        final_log_content = log_manager.get_logs(log_key)
        
        if rc == 0:
            status = "completed"
            print(f"Crawler {filename} executed successfully")
            trigger_related_tasks(crawler_id)
        else:
            status = "failed"
            log_manager.append(log_key, f"Process exited with code {rc}")
            final_log_content = log_manager.get_logs(log_key)
            print(f"Crawler {filename} failed")
            
    except Exception as e:
        err_msg = f"Exception: {str(e)}"
        log_manager.append(log_key, err_msg)
        final_log_content = log_manager.get_logs(log_key)
        print(f"Crawler execution failed: {e}")
        
    # 更新日志
    end_time = datetime.now().isoformat()
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("UPDATE crawler_logs SET end_time = ?, status = ?, log_content = ? WHERE id = ?", 
              (end_time, status, final_log_content, log_id))
    c.execute("UPDATE crawlers SET last_run = ?, status = ? WHERE id = ?", (end_time, status, crawler_id))
    conn.commit()
    conn.close()

def trigger_related_tasks(crawler_id):
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE trigger_crawler = ?", (crawler_id,))
    tasks = c.fetchall()
    conn.close()
    
    for task in tasks:
        print(f"Triggering related prediction task: {task['name']}")
        try:
            config_dict = json.loads(task['config'])
            task_config = TaskConfig(**config_dict)
            # 注意：这里直接调用 run_prediction_task 是同步的，会阻塞调度线程。
            # 应该使用 scheduler.add_job(func, 'date', run_date=now) 来一次性异步执行
            # 或者用 background task (但在 apscheduler 里没有 FastAPI context)
            # 最好的办法是起一个新线程
            from threading import Thread
            t = Thread(target=run_prediction_task, args=(task['id'], task_config, "crawler_trigger"))
            t.start()
        except Exception as e:
            print(f"Failed to trigger task {task['id']}: {e}")

@app.post("/api/crawler/{crawler_id}/run")
def manually_run_crawler(crawler_id: str, background_tasks: BackgroundTasks):
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("SELECT filename FROM crawlers WHERE id = ?", (crawler_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Crawler not found")
    
    filename = row[0]
    background_tasks.add_task(run_crawler_job, crawler_id, filename)
    return {"status": "triggered"}

@app.delete("/api/crawler/{crawler_id}")
def delete_crawler(crawler_id: str):
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("SELECT filename FROM crawlers WHERE id = ?", (crawler_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Crawler not found")
    
    filename = row[0]
    file_path = os.path.join(SCRIPT_DIR, filename)
    
    # Remove job from scheduler
    if scheduler.get_job(crawler_id):
        scheduler.remove_job(crawler_id)
        
    # Delete file
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass # Ignore file deletion errors
            
    # Delete from DB
    c.execute("DELETE FROM crawlers WHERE id = ?", (crawler_id,))
    c.execute("DELETE FROM crawler_logs WHERE crawler_id = ?", (crawler_id,))
    conn.commit()
    conn.close()
    
    return {"status": "deleted"}

# --- 任务管理 API ---

@app.post("/api/tasks")
def create_task(config: TaskConfig, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO tasks (id, name, db_file, table_name, config, status, created_at, cron_expression, trigger_crawler) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, config.taskName, config.dbFile, config.table, json.dumps(config.dict()), "pending", datetime.now().isoformat(), config.cron, config.trigger_crawler)
    )
    conn.commit()
    conn.close()
    
    # 立即执行一次
    background_tasks.add_task(run_prediction_task, task_id, config, "manual")
    return {"id": task_id, "status": "created"}

@app.get("/api/tasks", response_model=List[Task])
def get_tasks():
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/api/tasks/{task_id}/config", response_model=TaskConfig)
def get_task_config(task_id: str):
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT config FROM tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
        
    try:
        config_dict = json.loads(row['config'])
        return TaskConfig(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid config in DB: {e}")

@app.put("/api/tasks/{task_id}")
def update_task(task_id: str, config: TaskConfig):
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("SELECT id FROM tasks WHERE id = ?", (task_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
        
    c.execute("""
        UPDATE tasks 
        SET name = ?, db_file = ?, table_name = ?, config = ?, cron_expression = ?, trigger_crawler = ?
        WHERE id = ?
    """, (config.taskName, config.dbFile, config.table, json.dumps(config.dict()), config.cron, config.trigger_crawler, task_id))
    conn.commit()
    conn.close()
    return {"status": "updated", "id": task_id}

@app.post("/api/tasks/{task_id}/run")
def manually_run_task(task_id: str, background_tasks: BackgroundTasks):
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    task = c.fetchone()
    conn.close()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task['status'] == 'running':
         raise HTTPException(status_code=400, detail="Task is already running")

    try:
        config_dict = json.loads(task['config'])
        # 确保 config 符合当前模型结构
        # 有些旧任务可能缺少某些字段，这里做个简单兼容（如果需要）
        task_config = TaskConfig(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid task config: {e}")

    background_tasks.add_task(run_prediction_task, task_id, task_config, "manual_rerun")
    return {"status": "triggered"}

@app.get("/api/tasks/{task_id}/logs", response_model=List[PredictionLog])
def get_task_logs(task_id: str):
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM prediction_logs WHERE task_id = ? ORDER BY start_time DESC", (task_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- 结果查询 API (优化版) ---

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

@app.get("/api/tasks/{task_id}/result")
def get_task_result(task_id: str):
    try:
        conn = sqlite3.connect(METADATA_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        task = c.fetchone()
        conn.close()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
            
        config = json.loads(task['config'])
        
        conn = sqlite3.connect(RESULTS_DB)
        try:
            forecast_df = pd.read_sql_query(f"SELECT * FROM predictions WHERE task_id = '{task_id}'", conn)
            forecast_df = forecast_df.where(pd.notnull(forecast_df), None)
        except:
            forecast_df = pd.DataFrame()
            
        try:
            metrics_df = pd.read_sql_query(f"SELECT * FROM task_metrics WHERE task_id = '{task_id}'", conn)
            metrics_df = metrics_df.where(pd.notnull(metrics_df), None)
            metrics = {row['metric_name']: row['metric_value'] for _, row in metrics_df.iterrows()}
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            metrics = {}
        conn.close()
        
        db_path = os.path.join(DB_DIR, config['dbFile'])
        dimensions = {}
        history_data = []
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            table = config['table']
            
            # 获取历史数据 (全部，为了大屏展示)
            # 注意：数据量大时需优化
            history_df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5000", conn) # 临时限制
            conn.close()
            history_df = history_df.where(pd.notnull(history_df), None)

            
            # 提取维度信息
            group_cols = config['mapping'].get('group_cols', [])
            if not history_df.empty and group_cols:
                for col in group_cols:
                    if col in history_df.columns:
                        dimensions[col] = history_df[col].unique().tolist()
                        
            history_data = history_df.to_dict(orient="records")
        
        result = {
            "config": config,
            "history": history_data,
            "forecast": forecast_df.to_dict(orient="records"),
            "dimensions": dimensions, # 新增：返回维度元数据
            "metrics": metrics # 新增：返回评估指标
        }
        return sanitize_for_json(result)
    except Exception as e:
        print(f"Error in get_task_result: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 结果下载 API ---

@app.get("/api/tasks/{task_id}/download")
def download_task_result(task_id: str):
    conn = sqlite3.connect(RESULTS_DB)
    try:
        # 读取预测结果
        df = pd.read_sql_query(f"SELECT * FROM predictions WHERE task_id = '{task_id}'", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No results found")
        
    # 获取任务名称作为文件名
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    task_name = row[0] if row else "prediction_results"
    # 清理文件名非法字符
    task_name = "".join([c for c in task_name if c.isalnum() or c in (' ', '-', '_')]).strip()

    # 检查数据量 (Excel 限制约104万行，我们设为100万安全线)
    MAX_ROWS = 1000000
    
    if len(df) <= MAX_ROWS:
        # 单个文件
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Forecast')
        output.seek(0)
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={task_name}.xlsx"}
        )
    else:
        # 超过限制，拆分为多个文件并打包 ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            total_rows = len(df)
            num_parts = (total_rows // MAX_ROWS) + 1
            for i in range(num_parts):
                start_idx = i * MAX_ROWS
                end_idx = min((i + 1) * MAX_ROWS, total_rows)
                part_df = df.iloc[start_idx:end_idx]
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    part_df.to_excel(writer, index=False, sheet_name='Forecast')
                
                zip_file.writestr(f"{task_name}_part{i+1}.xlsx", excel_buffer.getvalue())
        
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={task_name}.zip"}
        )

# --- 真实训练逻辑 ---
def run_prediction_task(task_id: str, config: TaskConfig, trigger_type: str = "manual"):
    print(f"Starting DeepAR task {task_id}")
    
    # 初始化日志
    log_key = f"task:{task_id}"
    log_manager.clear(log_key)
    start_time = datetime.now().isoformat()
    
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("INSERT INTO prediction_logs (task_id, start_time, status, trigger_type) VALUES (?, ?, 'running', ?)", 
              (task_id, start_time, trigger_type))
    log_id = c.lastrowid
    conn.commit()
    conn.close()
    
    def logger(msg):
        print(f"[Task {task_id}] {msg}")
        log_manager.append(log_key, msg)
        
    logger(f"Starting Task {task_id} (Trigger: {trigger_type})")
    update_task_status(task_id, "running")
    
    status = "failed"
    final_log_content = ""
    
    try:
        try:
            from backend.app.model_engine import UnifiedForecastEngine
        except ImportError:
            from app.model_engine import UnifiedForecastEngine
            
        config.params['task_id'] = task_id
        config.params['models'] = config.models # 将 models 注入到 params 中，供引擎使用
        db_path = os.path.join(DB_DIR, config.dbFile)
        
        engine = UnifiedForecastEngine(
            db_path=db_path,
            table_name=config.table,
            mapping=config.mapping,
            params=config.params,
            logger=logger
        )
        
        results, metrics = engine.train_and_predict()
        engine.save_results(results, metrics)
        
        status = "completed"
        logger("Task completed successfully")
        update_task_status(task_id, "completed")
        
    except Exception as e:
        logger(f"Task failed: {e}")
        import traceback
        full_traceback = traceback.format_exc()
        logger(full_traceback)
        update_task_status(task_id, "failed", full_traceback)
        
    # 保存最终日志
    end_time = datetime.now().isoformat()
    final_log_content = log_manager.get_logs(log_key)
    
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    c.execute("UPDATE prediction_logs SET end_time = ?, status = ?, log_content = ? WHERE id = ?", 
              (end_time, status, final_log_content, log_id))
    conn.commit()
    conn.close()

def update_task_status(task_id: str, status: str, error_msg: str = None):
    conn = sqlite3.connect(METADATA_DB)
    c = conn.cursor()
    if error_msg:
        c.execute("UPDATE tasks SET status = ?, error_msg = ? WHERE id = ?", (status, error_msg, task_id))
    else:
        c.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)

