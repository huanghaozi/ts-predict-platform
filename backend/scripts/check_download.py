import requests

task_id = "dc10351c-fdc8-47f7-b2be-ef07fd87a422"
url = f"http://127.0.0.1:8100/api/tasks/{task_id}/download"

try:
    res = requests.get(url, stream=True)
    print(f"Status Code: {res.status_code}")
    print("Headers:")
    for k, v in res.headers.items():
        print(f"{k}: {v}")
except Exception as e:
    print(f"Error: {e}")

