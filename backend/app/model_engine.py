import pandas as pd
import numpy as np
import sqlite3
import json
import os
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from sklearn.preprocessing import LabelEncoder

# StatsForecast optional import
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS
    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False

class UnifiedForecastEngine:
    def __init__(self, db_path, table_name, mapping, params, logger=None):
        self.db_path = db_path
        self.table_name = table_name
        self.mapping = mapping
        self.params = params
        self.device = "cpu" # 强制 CPU
        self.logger_func = logger
        self.cardinalities = []
        self.series_info = [] 

    def log(self, msg):
        if self.logger_func:
            self.logger_func(msg)
        else:
            print(msg)

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        # 按时间排序读取
        time_col = self.mapping['time_col']
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name} ORDER BY {time_col}", conn)
        conn.close()
        
        df[time_col] = pd.to_datetime(df[time_col])
        return df

    def prepare_dataset_gluonts(self, df, freq='M', mode='train'):
        # mode: 'train', 'future', 'backtest'
        group_cols = self.mapping.get('group_cols', [])
        target_col = self.mapping['target_col']
        time_col = self.mapping['time_col']
        
        # 获取额外的静态分类列
        user_static_cols = self.mapping.get('static_cat_cols', [])
        all_static_cols = []
        seen = set()
        for c in group_cols + user_static_cols:
            if c not in seen:
                all_static_cols.append(c)
                seen.add(c)
        
        if freq == 'MS': freq = 'M'
        
        prediction_length = int(self.params.get('prediction_length', 12))

        # --- 静态特征编码 ---
        self.cardinalities = []
        encoded_cols_map = {}
        if all_static_cols:
            # self.log(f"Processing static features: {all_static_cols}")
            for col in all_static_cols:
                le = LabelEncoder()
                encoded_col_name = f"{col}_encoded"
                df[encoded_col_name] = le.fit_transform(df[col].astype(str))
                encoded_cols_map[col] = encoded_col_name
                
                card = len(le.classes_)
                self.cardinalities.append(card)
        
        # --- 动态特征处理 ---
        dynamic_real_cols = self.mapping.get('dynamic_real_cols', [])
        
        if not group_cols:
            groups = [('all', df)]
        else:
            groups = df.groupby(group_cols)

        data_list = []
        # Note: self.series_info might be overwritten if called multiple times, but OK for sequential runs
        local_series_info = [] 

        for name, group in groups:
            group = group.sort_values(time_col)
            target = group[target_col].values
            
            # Backtest: truncate target
            if mode == 'backtest':
                target = target[:-prediction_length]

            start = group[time_col].iloc[0]
            
            if isinstance(start, pd.Timestamp):
                start = pd.Period(start, freq)
            
            if isinstance(name, tuple):
                item_id = "|".join(map(str, name))
            else:
                item_id = str(name)

            data_entry = {
                FieldName.START: start,
                FieldName.TARGET: target,
                FieldName.ITEM_ID: item_id
            }
            
            # 注入静态特征
            if all_static_cols:
                feat_static_list = []
                for col in all_static_cols:
                    encoded_col = encoded_cols_map[col]
                    val = group[encoded_col].iloc[0]
                    feat_static_list.append(val)
                
                feat_static = np.array(feat_static_list, dtype=int)
                data_entry[FieldName.FEAT_STATIC_CAT] = feat_static

            # 注入动态特征
            if dynamic_real_cols:
                # 1. 获取历史部分 (shape: num_features x history_length)
                feat_dynamic_hist = group[dynamic_real_cols].values.T.astype(np.float32)
                
                if mode == 'future':
                    # 2. 构造未来部分 (简单方案: 用最后一个值向前填充)
                    last_values = feat_dynamic_hist[:, -1].reshape(-1, 1)
                    feat_dynamic_future = np.repeat(last_values, prediction_length, axis=1)
                    
                    # 3. 拼接
                    feat_dynamic = np.concatenate([feat_dynamic_hist, feat_dynamic_future], axis=1)
                elif mode == 'backtest':
                    feat_dynamic = feat_dynamic_hist
                else: # train
                    feat_dynamic = feat_dynamic_hist
                    
                data_entry[FieldName.FEAT_DYNAMIC_REAL] = feat_dynamic

            data_list.append(data_entry)
            
            local_series_info.append({
                "item_id": item_id, 
                "group_vals": name 
            })
        
        # Update global series info only if relevant (e.g. training phase)
        if mode == 'train':
            self.series_info = local_series_info

        return ListDataset(data_list, freq=freq)

    def run_deepar(self, df):
        freq = self.params.get('freq', 'M')
        if freq == 'MS': freq = 'M'
        
        prediction_length = int(self.params.get('prediction_length', 12))
        context_length = int(self.params.get('context_length', 24))
        epochs = int(self.params.get('epochs', 5))

        # 1. Training
        train_ds = self.prepare_dataset_gluonts(df, freq, mode='train')

        num_static_cat = len(self.cardinalities)
        num_dynamic_real = len(self.mapping.get('dynamic_real_cols', []))

        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer_kwargs={"max_epochs": epochs, "accelerator": "cpu"},
            num_feat_static_cat=num_static_cat,
            cardinality=self.cardinalities if self.cardinalities else None,
            num_feat_dynamic_real=num_dynamic_real
        )

        self.log(f"DeepAR: 开始训练 (Static Feats: {num_static_cat}, Dynamic Feats: {num_dynamic_real})...")
        predictor = estimator.train(train_ds)
        
        # 2. Backtest Evaluation
        self.log("DeepAR: 开始回测评估...")
        backtest_ds = self.prepare_dataset_gluonts(df, freq, mode='backtest')
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=backtest_ds,
            predictor=predictor,
            num_samples=100
        )
        
        forecasts_eval = list(forecast_it)
        tss_eval = list(ts_it)
        
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, _ = evaluator(tss_eval, forecasts_eval)
        
        # Rename metrics to include model name
        model_metrics = {f"DeepAR:{k}": v for k, v in agg_metrics.items()}
        self.log(f"DeepAR 评估完成: RMSE={agg_metrics['RMSE']:.4f}")

        # 3. Future Prediction
        self.log("DeepAR: 开始未来预测...")
        predict_ds = self.prepare_dataset_gluonts(df, freq, mode='future')
        forecasts = list(predictor.predict(predict_ds))
        
        results = []
        for i, forecast in enumerate(forecasts):
            info = self.series_info[i]
            
            p10 = forecast.quantile(0.1)
            p50 = forecast.quantile(0.5)
            p90 = forecast.quantile(0.9)
            
            start_ts = forecast.start_date.to_timestamp() if hasattr(forecast.start_date, 'to_timestamp') else forecast.start_date
            
            dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(
                start=start_ts, 
                periods=prediction_length, 
                freq=freq
            )]
            
            for j, date in enumerate(dates):
                results.append({
                    "task_id": self.params.get('task_id'),
                    "item_id": info['item_id'],
                    "date": date,
                    "p10": float(p10[j]),
                    "p50": float(p50[j]),
                    "p90": float(p90[j]),
                    "type": "prediction",
                    "model": "DeepAR"
                })
                
        return results, model_metrics

    def run_statsforecast(self, df, model_names):
        if not HAS_STATSFORECAST:
            self.log("Error: StatsForecast not installed.")
            return [], {}

        self.log(f"StatsForecast: 开始训练 {model_names}...")
        
        # Prepare DataFrame for StatsForecast (unique_id, ds, y)
        group_cols = self.mapping.get('group_cols', [])
        target_col = self.mapping['target_col']
        time_col = self.mapping['time_col']
        
        # Construct unique_id
        if not group_cols:
            df['unique_id'] = 'all'
        else:
            # Optimized implementation for creating unique_id
            if len(group_cols) == 1:
                df['unique_id'] = df[group_cols[0]].astype(str)
            else:
                df['unique_id'] = df[group_cols].astype(str).agg('|'.join, axis=1)
        
        sf_df = df[[ 'unique_id', time_col, target_col ]].rename(columns={time_col: 'ds', target_col: 'y'})
        
        # Determine frequency
        freq = self.params.get('freq', 'M')
        if freq == 'M': freq = 'M' # StatsForecast uses Pandas aliases
        
        # Auto-detect Month Start vs Month End mismatch
        # If data is 2020-01-01 (MS) and freq='M', StatsForecast might output 2020-01-31
        # leading to merge failure (empty metrics)
        dates = pd.to_datetime(sf_df['ds']).sort_values().unique()
        if len(dates) > 1:
            # Check if day is 1
            is_start_of_month = all(pd.Timestamp(d).day == 1 for d in dates[:5])
            if is_start_of_month and freq == 'M':
                self.log("Detected Month Start data but freq='M'. Converting freq to 'MS' for StatsForecast compatibility.")
                freq = 'MS'

        prediction_length = int(self.params.get('prediction_length', 12))
        
        # Initialize models
        models = []
        for m in model_names:
            if m == 'ARIMA':
                models.append(AutoARIMA(season_length=12 if freq in ['M', 'MS'] else 1))
            elif m == 'ETS':
                models.append(AutoETS(season_length=12 if freq in ['M', 'MS'] else 1))
        
        sf = StatsForecast(models=models, freq=freq, n_jobs=1)
        
        # Fit and Predict
        # Backtest for metrics
        # StatsForecast has cross_validation, but let's stick to simple train/test split logic for consistency
        # Note: StatsForecast models are local, so "backtest" is just training on truncated data.
        
        self.log("StatsForecast: 开始回测评估...")
        # We do a simple holdout evaluation
        train_df = sf_df.groupby('unique_id').apply(lambda x: x.iloc[:-prediction_length]).reset_index(drop=True)
        test_df = sf_df.groupby('unique_id').apply(lambda x: x.iloc[-prediction_length:]).reset_index(drop=True)
        
        sf.fit(train_df)
        forecast_backtest = sf.predict(h=prediction_length, level=[80]) # 80% confidence -> p10, p90
        
        # Rename columns to match user-friendly model names (AutoARIMA -> ARIMA)
        name_map = {'AutoARIMA': 'ARIMA', 'AutoETS': 'ETS'}
        
        # 1. Rename backtest columns
        cols = forecast_backtest.columns
        new_cols = {}
        for c in cols:
            for k, v in name_map.items():
                if k in c:
                    # Replace 'AutoARIMA' with 'ARIMA' in column name
                    new_cols[c] = c.replace(k, v)
        if new_cols:
            forecast_backtest = forecast_backtest.rename(columns=new_cols)

        # Calculate metrics
        # We need to merge with test_df to compare
        forecast_backtest = forecast_backtest.reset_index()
        merged = forecast_backtest.merge(test_df, on=['unique_id', 'ds'], how='inner')
        
        metrics = {}
        for m in model_names:
            # Calculate simple RMSE/MAPE for this model
            if m not in merged.columns:
                self.log(f"Warning: Column {m} not found in forecast results. Available columns: {merged.columns}")
                continue
                
            y_true = merged['y'].values
            y_pred = merged[m].values
            
            mse = np.mean((y_true - y_pred)**2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) # Avoid div by zero
            
            metrics[f"{m}:RMSE"] = rmse
            metrics[f"{m}:MAPE"] = mape
            
        # Future Prediction
        self.log("StatsForecast: 开始未来预测...")
        sf.fit(sf_df) # Retrain on full data
        forecast_future = sf.predict(h=prediction_length, level=[80])
        
        # 2. Rename future columns
        cols = forecast_future.columns
        new_cols = {}
        for c in cols:
            for k, v in name_map.items():
                if k in c:
                    new_cols[c] = c.replace(k, v)
        if new_cols:
            forecast_future = forecast_future.rename(columns=new_cols)
            
        forecast_future = forecast_future.reset_index()
        
        results = []
        for _, row in forecast_future.iterrows():
            item_id = row['unique_id']
            date = row['ds'].strftime('%Y-%m-%d')
            
            for m in model_names:
                if m not in row:
                    continue
                    
                # StatsForecast output columns: unique_id, ds, ARIMA, ARIMA-lo-80, ARIMA-hi-80
                p50 = row[m]
                p10 = row.get(f"{m}-lo-80", p50) # Fallback
                p90 = row.get(f"{m}-hi-80", p50)
                
                results.append({
                    "task_id": self.params.get('task_id'),
                    "item_id": item_id,
                    "date": date,
                    "p10": float(p10),
                    "p50": float(p50),
                    "p90": float(p90),
                    "type": "prediction",
                    "model": m
                })

        return results, metrics

    def train_and_predict(self):
        df = self.load_data()
        models_to_run = self.params.get('models', ['DeepAR'])
        all_results = []
        all_metrics = {}
        
        if isinstance(models_to_run, str): # Handle single string case if any
            models_to_run = [models_to_run]

        if 'DeepAR' in models_to_run:
             try:
                 res, met = self.run_deepar(df.copy())
                 all_results.extend(res)
                 all_metrics.update(met)
             except Exception as e:
                 self.log(f"DeepAR failed: {e}")
                 import traceback
                 self.log(traceback.format_exc())

        stats_models = [m for m in models_to_run if m in ['ARIMA', 'ETS']]
        if stats_models:
             try:
                 res, met = self.run_statsforecast(df.copy(), stats_models)
                 all_results.extend(res)
                 all_metrics.update(met)
             except Exception as e:
                 self.log(f"StatsForecast failed: {e}")
                 import traceback
                 self.log(traceback.format_exc())
                 
        if not all_results:
             raise Exception("No models executed successfully.")

        return all_results, all_metrics

    def save_results(self, results, metrics=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res_db_path = os.path.join(base_dir, "data", "results.db")
        os.makedirs(os.path.dirname(res_db_path), exist_ok=True)
        
        conn = sqlite3.connect(res_db_path)
        c = conn.cursor()
        
        # Check and migrate schema for 'model' column
        try:
            c.execute("SELECT model FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            self.log("Migrating database: Adding 'model' column to predictions table...")
            try:
                c.execute("ALTER TABLE predictions ADD COLUMN model TEXT")
                c.execute("UPDATE predictions SET model = 'DeepAR' WHERE model IS NULL") # Default old records
            except Exception as e:
                self.log(f"Migration failed: {e}")

        task_id = self.params.get('task_id')
        
        # Clear old data for this task to avoid mixing results from different runs/models
        try:
            c.execute("DELETE FROM predictions WHERE task_id = ?", (task_id,))
            c.execute("DELETE FROM task_metrics WHERE task_id = ?", (task_id,))
        except Exception as e:
            self.log(f"Warning: Failed to clear old data: {e}")

        if results:
            df_res = pd.DataFrame(results)
            df_res.to_sql('predictions', conn, if_exists='append', index=False)
            
        if metrics:
            metrics_list = []
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    metrics_list.append({
                        "task_id": task_id,
                        "metric_name": k,
                        "metric_value": float(v) if isinstance(v, (int, float)) else v
                    })
            
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                df_metrics.to_sql('task_metrics', conn, if_exists='append', index=False)
                
        conn.commit()
        conn.close()
