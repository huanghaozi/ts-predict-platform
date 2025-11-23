import requests
import json

url = "http://localhost:8100/api/tasks"
payload = {
  "taskName": "Test Task via API",
  "dbFile": "mock_shipping.db",
  "table": "shipping_data",
  "mapping": {
    "time_col": "start_date",
    "target_col": "volume",
    "group_cols": ["product_name"],
    "static_feat_cols": [],
    "dynamic_feat_cols": ["gdp", "sentiment"]
  },
  "models": ["DeepAR"],
  "params": {
    "prediction_length": 12,
    "freq": "M",
    "context_length": 24,
    "epochs": 1 # 设为 1 epoch 加快测试速度
  }
}

try:
    res = requests.post(url, json=payload)
    print(res.status_code)
    print(res.json())
except Exception as e:
    print(e)

