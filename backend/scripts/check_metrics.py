import sqlite3
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "results.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

task_id = "8e42b055-7767-40a8-8dfd-e946851c2989"
try:
    cursor.execute(f"SELECT * FROM task_metrics WHERE task_id = '{task_id}'")
    metrics = cursor.fetchall()
    print(f"Metrics for task {task_id}:")
    for m in metrics:
        print(m)
except Exception as e:
    print(f"Error: {e}")

conn.close()

