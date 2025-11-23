import requests
import time
import json
import sys

BASE_URL = "http://localhost:8100/api"

def test_e2e():
    print("1. Testing File List...")
    res = requests.get(f"{BASE_URL}/files/db")
    files = res.json()
    print(f"Files: {files}")
    
    db_file = next((f for f in files if f['name'] == 'mock_shipping.db'), None)
    if not db_file:
        print("Error: mock_shipping.db not found!")
        return

    print("\n2. Creating Task...")
    payload = {
        "taskName": "E2E Test Task",
        "dbFile": "mock_shipping.db",
        "table": "shipping_data",
        "mapping": {
            "time_col": "start_date",
            "target_col": "volume",
            "group_cols": ["product_name"]
        },
        "models": ["DeepAR"],
        "params": {
            "prediction_length": 6,
            "freq": "MS", # 月初
            "context_length": 12,
            "epochs": 1
        }
    }
    
    res = requests.post(f"{BASE_URL}/tasks", json=payload)
    if res.status_code != 200:
        print(f"Create Task Failed: {res.text}")
        return
        
    task_id = res.json()['id']
    print(f"Task Created: {task_id}")
    
    print("\n3. Waiting for completion...")
    for i in range(30): # 等待 30秒
        res = requests.get(f"{BASE_URL}/tasks")
        tasks = res.json()
        current_task = next((t for t in tasks if t['id'] == task_id), None)
        
        status = current_task['status']
        print(f"Status: {status}")
        
        if status == 'completed':
            print("Task Completed!")
            break
        elif status == 'failed':
            print(f"Task Failed! Error: {current_task.get('error_msg')}")
            return
            
        time.sleep(2)
        
    print("\n4. Fetching Results...")
    res = requests.get(f"{BASE_URL}/tasks/{task_id}/result")
    result = res.json()
    
    forecasts = result.get('forecast', [])
    print(f"Forecast count: {len(forecasts)}")
    if len(forecasts) > 0:
        print("Sample forecast:", forecasts[0])
        print("Test PASSED!")
    else:
        print("No forecasts found. Test FAILED.")

if __name__ == "__main__":
    try:
        test_e2e()
    except Exception as e:
        print(f"Test Error: {e}")

