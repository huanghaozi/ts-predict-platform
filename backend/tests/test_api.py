import pytest
from fastapi.testclient import TestClient
import os
import shutil
import sys

# 确保能导入 backend 模块
# 假设测试运行在项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app, DB_DIR, METADATA_DB, init_dbs

client = TestClient(app)

# --- Fixtures ---

@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    """设置测试环境：清理并重建测试目录"""
    print(f"\n[Setup] DB_DIR: {DB_DIR}")
    
    # 备份原有的 metadata (可选，这里为了简单直接用)
    # 或者我们可以临时修改 DB_DIR 为 test_db_dir
    # 但由于 main.py 里 DB_DIR 是全局变量，直接修改比较麻烦。
    # 我们选择：确保 mock_shipping.db 存在
    
    os.makedirs(DB_DIR, exist_ok=True)
    
    # 创建一个测试用的空 DB 文件
    test_db_path = os.path.join(DB_DIR, "test_shipping.db")
    with open(test_db_path, "w") as f:
        f.write("dummy content") # 这不是合法的 SQLite，但足以测试文件列表 API
        
    yield
    
    # Teardown
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    print("\n[Teardown] Cleaned up test files")

# --- Tests ---

def test_list_db_files():
    """测试获取 DB 文件列表接口"""
    response = client.get("/api/files/db")
    assert response.status_code == 200
    data = response.json()
    print(f"\nResponse Data: {data}")
    
    assert isinstance(data, list)
    # 应该至少包含我们在 setup 里创建的 test_shipping.db
    # 也可能包含之前生成的 mock_shipping.db
    filenames = [f['name'] for f in data]
    assert "test_shipping.db" in filenames
    
    # 检查路径是否为绝对路径 (根据之前的修改)
    for f in data:
        assert os.path.isabs(f['path'])

def test_upload_db_file():
    """测试上传文件"""
    # 创建一个临时文件
    files = {'file': ('upload_test.db', b'sqlite content', 'application/octet-stream')}
    response = client.post("/api/files/upload", files=files)
    
    assert response.status_code == 200
    assert response.json()['status'] == 'uploaded'
    
    # 验证是否在列表中
    response = client.get("/api/files/db")
    filenames = [f['name'] for f in response.json()]
    assert "upload_test.db" in filenames
    
    # 清理
    os.remove(os.path.join(DB_DIR, "upload_test.db"))

def test_get_tables_failure():
    """测试获取不存在的 DB 表结构"""
    response = client.get("/api/db/non_existent.db/tables")
    assert response.status_code == 404

def test_task_creation_flow():
    """测试任务创建流程"""
    # 1. 需要一个合法的 SQLite 文件
    # 为了测试真实逻辑，我们需要真正的 SQLite 文件，不能用 dummy content
    # 我们可以调用 backend/scripts/mock_data_generator.py 的逻辑生成一个
    
    from backend.scripts.mock_data_generator import generate_mock_data
    real_db_path = os.path.join(DB_DIR, "pytest_mock.db")
    generate_mock_data(real_db_path)
    
    # 2. 创建任务 payload
    payload = {
        "taskName": "Pytest Task",
        "dbFile": "pytest_mock.db",
        "table": "shipping_data",
        "mapping": {
            "time_col": "start_date",
            "target_col": "volume",
            "group_cols": ["product_name"]
        },
        "models": ["DeepAR"],
        "params": {
            "prediction_length": 6,
            "freq": "M",
            "epochs": 1
        }
    }
    
    response = client.post("/api/tasks", json=payload)
    assert response.status_code == 200
    task_id = response.json()['id']
    assert task_id is not None
    
    # 3. 检查任务列表
    response = client.get("/api/tasks")
    tasks = response.json()
    assert any(t['id'] == task_id for t in tasks)
    
    # 4. 清理
    if os.path.exists(real_db_path):
        os.remove(real_db_path)

if __name__ == "__main__":
    # 允许直接运行脚本
    sys.exit(pytest.main(["-v", __file__]))

