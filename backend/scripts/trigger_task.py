import requests
import json

url = "http://localhost:8100/api/tasks"
payload = {
    "taskName": "Browser Test Task 2",
    "dbFile": "mock_shipping.db",
    "table": "shipping_data",
    "mapping": {
        "time_col": "start_date",
        "target_col": "volume",
        "group_cols": ["product_name"]
    },
    "models": ["DeepAR"],
    "params": {
        "prediction_length": 12,
        "freq": "M",
        "epochs": 1
    }
}

res = requests.post(url, json=payload)
print(res.status_code)
print(res.json())

