import requests
import json

task_id = "dc10351c-fdc8-47f7-b2be-ef07fd87a422"
res = requests.get(f"http://127.0.0.1:8100/api/tasks/{task_id}/logs")
logs = res.json()

print(f"Logs for {task_id}:")
for log in logs:
    content = log.get("log_content", "")
    print(content)
    
    if "Static Feats:" in content:
        print("CONFIRMED: Static Features used.")
    if "Dynamic Feats: 4" in content: # We passed 4 dynamic cols
        print("CONFIRMED: Dynamic Features used.")

