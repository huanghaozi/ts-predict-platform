import requests
import json
import time
import sys

url = "http://127.0.0.1:8100/api/tasks"

config = {
    "taskName": "API-Test-Full-Features",
    "dbFile": "mock_shipping.db",
    "table": "shipping_data",
    "mapping": {
        "time_col": "start_date",
        "target_col": "volume",
        "group_cols": ["category_large", "category_mid", "product_name"],
        "static_cat_cols": ["manager_id"],
        "dynamic_real_cols": ["gdp", "sentiment", "policy_strength", "attention"]
    },
    "models": ["DeepAR"],
    "params": {
        "prediction_length": 12,
        "context_length": 24,
        "epochs": 5,
        "freq": "M"
    }
}

print("Creating task...")
try:
    res = requests.post(url, json=config)
    if res.status_code != 200:
        print(f"Failed to create task: {res.text}")
        sys.exit(1)

    task_id = res.json()["id"]
    print(f"Task created: {task_id}")

    # Poll status
    while True:
        res = requests.get("http://127.0.0.1:8100/api/tasks")
        tasks = res.json()
        my_task = next((t for t in tasks if t["id"] == task_id), None)
        
        if not my_task:
            print("Task not found?")
            break
            
        status = my_task["status"]
        print(f"Status: {status}")
        
        if status == "completed":
            print("Task completed successfully!")
            break
        elif status == "failed":
            print(f"Task failed: {my_task.get('error_msg')}")
            break
            
        time.sleep(2)

    # Check logs for feature confirmation
    print("Checking logs...")
    res = requests.get(f"http://127.0.0.1:8100/api/tasks/{task_id}/logs")
    logs = res.json()
    
    found_static = False
    found_dynamic = False
    
    for log in logs:
        content = log.get("log_content", "")
        # The content might be inside the log entry or streaming log content
        # My API returns list of PredictionLog objects. 
        # The `log_content` field stores the final accumulated log.
        # Also checking live logs? No, task is done.
        if content:
            print(f"Log content: {content[:200]}...") # Print preview
            if "Static Feats:" in content:
                found_static = True
                print("CONFIRMED: Static Features used.")
            if "Dynamic Feats: True" in content:
                found_dynamic = True
                print("CONFIRMED: Dynamic Features used.")
                
    if found_static and found_dynamic:
        print("SUCCESS: Both Static and Dynamic features were utilized.")
    else:
        print("WARNING: Did not find explicit confirmation in logs.")

except Exception as e:
    print(f"Error: {e}")

