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

    def fill_missing_periods(self, df):
        """
        填充缺失的时间周期，将缺失值填充为0
        适用于发货量、销售量等场景
        """
        group_cols = self.mapping.get('group_cols', [])
        target_col = self.mapping['target_col']
        time_col = self.mapping['time_col']
        freq = self.params.get('freq', 'M')
        
        if not group_cols:
            # 无分组：直接对整个时间序列填充
            df = df.sort_values(time_col)
            
            # 创建完整的时间范围
            full_date_range = pd.date_range(
                start=df[time_col].min(),
                end=df[time_col].max(),
                freq=freq
            )
            
            # 创建完整的数据框
            full_df = pd.DataFrame({time_col: full_date_range})
            
            # 合并并填充
            result = full_df.merge(df, on=time_col, how='left')
            result[target_col] = result[target_col].fillna(0)
            
            # 填充其他列（静态特征用前向填充，动态特征用0）
            static_cols = self.mapping.get('static_cat_cols', [])
            dynamic_cols = self.mapping.get('dynamic_real_cols', [])
            
            for col in static_cols:
                if col in result.columns:
                    result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
            
            for col in dynamic_cols:
                if col in result.columns:
                    result[col] = result[col].fillna(0)
            
            return result
        else:
            # 有分组：对每个组分别填充
            all_groups = []
            
            for name, group in df.groupby(group_cols):
                group = group.sort_values(time_col)
                
                # 创建该组的完整时间范围
                full_date_range = pd.date_range(
                    start=group[time_col].min(),
                    end=group[time_col].max(),
                    freq=freq
                )
                
                # 创建完整的数据框
                full_df = pd.DataFrame({time_col: full_date_range})
                
                # 添加分组列
                if isinstance(name, tuple):
                    for i, col in enumerate(group_cols):
                        full_df[col] = name[i]
                else:
                    full_df[group_cols[0]] = name
                
                # 合并并填充
                result = full_df.merge(group, on=[time_col] + group_cols, how='left')
                result[target_col] = result[target_col].fillna(0)
                
                # 填充其他列
                static_cols = self.mapping.get('static_cat_cols', [])
                dynamic_cols = self.mapping.get('dynamic_real_cols', [])
                
                for col in static_cols:
                    if col in result.columns and col not in group_cols:
                        result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
                
                for col in dynamic_cols:
                    if col in result.columns:
                        result[col] = result[col].fillna(0)
                
                all_groups.append(result)
            
            return pd.concat(all_groups, ignore_index=True)


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

        skipped_count = 0
        total_count = 0
        filled_count = 0
        
        # 获取缺失数据处理策略
        missing_strategy = self.params.get('missing_data_strategy', 'skip')
        
        for name, group in groups:
            total_count += 1
            group = group.sort_values(time_col)
            target = group[target_col].values
            
            # 检查数据长度是否足够
            context_length = int(self.params.get('context_length', 24))
            min_required = prediction_length + context_length
            if mode == 'backtest':
                # 回测需要额外的 prediction_length 用于评估
                min_required = 2 * prediction_length + context_length
            
            # 长度检查和处理
            if len(target) < min_required:
                if missing_strategy == 'fill_zero':
                    # 策略：填充0到最小要求长度
                    filled_count += 1
                    shortage = min_required - len(target)
                    
                    if filled_count <= 3:  # 只显示前3个填充警告
                        self.log(f"📝 填充序列 '{name}': 原长度 {len(target)}, 填充 {shortage} 个0")
                    
                    # 在前面填充0（假设缺失的是早期数据）
                    target = np.concatenate([np.zeros(shortage), target])
                    
                    # 同时需要扩展时间索引
                    start = group[time_col].iloc[0]
                    # 计算填充后的新起始时间
                    if freq == 'M':
                        freq_pd = 'MS'
                    else:
                        freq_pd = freq
                    new_start = pd.date_range(end=start, periods=shortage + 1, freq=freq_pd)[0]
                    start = new_start
                    
                    # 动态特征也需要填充
                    if dynamic_real_cols:
                        original_feat = group[dynamic_real_cols].values.T.astype(np.float32)
                        padded_feat = np.zeros((original_feat.shape[0], shortage), dtype=np.float32)
                        group_feat_dynamic = np.concatenate([padded_feat, original_feat], axis=1)
                    
                else:
                    # 策略：跳过
                    skipped_count += 1
                    if skipped_count <= 5:  # 只显示前5个警告
                        self.log(f"⚠️  跳过序列 '{name}': 长度 {len(target)} < 最小要求 {min_required}")
                    continue
            else:
                # 长度足够，正常处理
                start = group[time_col].iloc[0]
                if dynamic_real_cols:
                    group_feat_dynamic = group[dynamic_real_cols].values.T.astype(np.float32)
            
            # Backtest: truncate target
            if mode == 'backtest':
                target = target[:-prediction_length]
            
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
                # 使用前面准备的 group_feat_dynamic，如果不存在则现在读取
                if 'group_feat_dynamic' not in locals():
                    group_feat_dynamic = group[dynamic_real_cols].values.T.astype(np.float32)
                
                if mode == 'future':
                    # 2. 构造未来部分 (简单方案: 用最后一个值向前填充)
                    last_values = group_feat_dynamic[:, -1].reshape(-1, 1)
                    feat_dynamic_future = np.repeat(last_values, prediction_length, axis=1)
                    
                    # 3. 拼接
                    feat_dynamic = np.concatenate([group_feat_dynamic, feat_dynamic_future], axis=1)
                elif mode == 'backtest':
                    # 回测模式：动态特征也需要截断
                    feat_dynamic = group_feat_dynamic[:, :-prediction_length]
                else: # train
                    feat_dynamic = group_feat_dynamic
                    
                data_entry[FieldName.FEAT_DYNAMIC_REAL] = feat_dynamic

            data_list.append(data_entry)
            
            local_series_info.append({
                "item_id": item_id, 
                "group_vals": name 
            })
        
        # 输出统计信息
        if skipped_count > 0 or filled_count > 0:
            parts = [f"{mode.upper()} 数据统计: 总序列 {total_count}, 有效 {len(data_list)}"]
            if skipped_count > 0:
                parts.append(f"跳过 {skipped_count}")
            if filled_count > 0:
                parts.append(f"填充 {filled_count}")
            self.log(f"📊 {', '.join(parts)}")
        
        # 确保至少有一些有效序列
        if len(data_list) == 0:
            raise ValueError(f"所有时间序列都被过滤掉了！模式: {mode}, 最小长度要求: {min_required} 个时间点")

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
        
        if not forecasts_eval:
            self.log("DeepAR Evaluation: No forecasts generated (data might be too short for backtest). Skipping metrics.")
            model_metrics = {}
        else:
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
