import sqlite3
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "results.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

task_id = "e65ceb4f-e8bd-4358-b5ef-6fb142e8cd8d"
cursor.execute(f"SELECT DISTINCT item_id FROM predictions WHERE task_id = '{task_id}'")
item_ids = cursor.fetchall()

print(f"Item IDs for task {task_id}:")
for item in item_ids:
    print(item[0])

conn.close()

