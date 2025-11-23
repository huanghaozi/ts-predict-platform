import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import sqlite3

def generate_mock_data(output_path="backend/data/db/mock_shipping.db"):
    print("Starting mock data generation (English)...")
    
    # 1. Define Dimensions (English)
    categories = {
        "Electronics": {
            "Mobile": ["Flagship-A", "Budget-B", "Foldable-C"],
            "Laptop": ["Gaming-X", "Ultrabook-Y"],
        },
        "HomeAppliances": {
            "Kitchen": ["SmartCooker", "AirFryer"],
            "Living": ["RobotVacuum", "AirPurifier"],
        }
    }
    
    managers = ["M001", "M002", "M003", "M004"]
    
    # 2. Time Range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='MS') 
    
    data_rows = []
    
    # 3. Base Macro Data
    gdp_base = np.linspace(100, 120, len(dates)) + np.random.normal(0, 0.5, len(dates))
    sentiment_base = 0.5 + 0.3 * np.sin(np.linspace(0, 8*np.pi, len(dates)))
    
    for i, date in enumerate(dates):
        curr_gdp = gdp_base[i]
        curr_sentiment = sentiment_base[i]
        policy_strength = round(random.uniform(0.1, 1.0), 2)
        
        for cat_large, mid_cats in categories.items():
            for cat_mid, products in mid_cats.items():
                for product_name in products:
                    mgr_idx = hash(product_name) % len(managers)
                    manager_id = managers[mgr_idx]
                    
                    base_vol = (hash(product_name) % 500) + 500 
                    trend = i * 2
                    month = date.month
                    seasonality = base_vol * 0.3 * np.sin(2 * np.pi * month / 12)
                    macro_impact = (curr_gdp - 100) * 10 + (curr_sentiment * 200)
                    noise = np.random.normal(0, base_vol * 0.1)
                    
                    shipment_volume = base_vol + trend + seasonality + macro_impact + noise
                    shipment_volume = max(0, int(shipment_volume))
                    
                    attention = int(shipment_volume * 0.1 + random.uniform(0, 100))
                    
                    month_end = date + pd.offsets.MonthEnd(0)
                    
                    row = {
                        "category_large": cat_large,
                        "category_mid": cat_mid,
                        "product_name": product_name,
                        "start_date": date.strftime("%Y-%m-%d"),
                        "end_date": month_end.strftime("%Y-%m-%d"),
                        "volume": shipment_volume,
                        "gdp": round(curr_gdp, 2),
                        "sentiment": round(curr_sentiment, 2),
                        "policy_strength": policy_strength,
                        "attention": attention,
                        "manager_id": manager_id
                    }
                    data_rows.append(row)
                    
    df = pd.DataFrame(data_rows)
    
    # Use absolute path for safety
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/scripts -> backend
    db_path = os.path.join(base_dir, "data", "db", "mock_shipping.db")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    df.to_sql('shipping_data', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"Mock data generated at: {db_path}, table: shipping_data, rows: {len(df)}")

if __name__ == "__main__":
    generate_mock_data()
