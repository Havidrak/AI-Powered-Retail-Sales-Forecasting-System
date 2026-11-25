import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. HYBRID FORECASTING ENGINE ---

def get_forecast_details(data, periods=30):
    """
    Forecasting using Random Forest & XGBoost ensemble.
    Returns total predicted sum and daily prediction list.
    """
    try:
        if len(data) < 7:
            mean_val = np.mean(data)
            return mean_val * periods, [mean_val] * periods

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        window_size = min(7, len(data) // 2)
        X, y = [], []

        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X, y)

        # Train XGBoost
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=1)
        xgb_model.fit(X, y)

        # Generate Forecast
        current_sequence = scaled_data[-window_size:].flatten()
        predictions = []

        for _ in range(periods):
            input_seq = current_sequence.reshape(1, -1)
            pred_rf = rf_model.predict(input_seq)[0]
            pred_xgb = xgb_model.predict(input_seq)[0]
            avg_pred = (pred_rf + pred_xgb) / 2
            
            predictions.append(avg_pred)
            current_sequence = np.append(current_sequence[1:], avg_pred)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return max(0, float(predictions.sum())), predictions.tolist()

    except Exception as e:
        print(f"Forecast error: {e}")
        return np.mean(data) * periods, [np.mean(data)] * periods

# --- 2. CATEGORY CLASSIFICATION ---

def assign_category(product_name):
    p = str(product_name).lower()
    if any(x in p for x in ['screen', 'cover', 'case', 'protector', 'holder', 'mount']): return 'Accessories'
    if any(x in p for x in ['cable', 'charger', 'power', 'bank', 'adapter', 'battery']): return 'Charging'
    if any(x in p for x in ['chair', 'desk', 'table', 'sofa', 'cabinet', 'shelf', 'lamp']): return 'Furniture'
    if any(x in p for x in ['clothing', 'shirt', 'pants', 'dress', 'shoe', 'beauty', 'makeup', 'skin', 'care']): return 'Lifestyle'
    if any(x in p for x in ['laptop', 'computer', 'monitor', 'printer', 'tablet', 'phone', 'mobile']): return 'Devices'
    if any(x in p for x in ['usb', 'card', 'drive', 'memory', 'disk', 'mouse', 'keyboard', 'headphone']): return 'Peripherals'
    if 'electronic' in p: return 'Electronics'
    return 'Other'

# --- 3. MAIN STOCK PROCESSING ---

def process_restock_data(filepath, default_stock=100):
    try:
        # Load Data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Identify Columns
        cols = df.columns.str.lower()
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        qty_col = next((c for c in df.columns if any(x in c.lower() for x in ['qty', 'quantity', 'units'])), None)
        prod_col = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'item', 'sku'])), None)

        if not all([date_col, qty_col, prod_col]):
            return None, "Missing columns. Need: Date, Quantity, Product Name"

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, qty_col, prod_col])

        # Analyze Top 50 Products
        top_products = df.groupby(prod_col)[qty_col].sum().nlargest(50).index.tolist()
        report = []

        for product in top_products:
            p_data = df[df[prod_col] == product].copy()
            daily_demand = p_data.set_index(date_col)[qty_col].resample('D').sum().fillna(0)

            # Historical Data for Chart
            history_slice = daily_demand.tail(30)
            history_dates = [d.strftime('%Y-%m-%d') for d in history_slice.index]
            history_values = history_slice.values.tolist()

            # Generate AI Forecast
            total_forecast, daily_predictions = get_forecast_details(daily_demand.values, periods=30)
            last_date = daily_demand.index.max()
            future_dates = [(last_date + timedelta(days=x + 1)).strftime('%Y-%m-%d') for x in range(30)]

            # Inventory Calculations
            avg_daily = np.mean(daily_demand.values)
            std_dev = np.std(daily_demand.values)
            lead_time = 7
            daily_forecast_rate = total_forecast / 30 if total_forecast > 0 else avg_daily

            safety_stock = int(np.ceil(std_dev * 1.65 * np.sqrt(lead_time)))
            rop = int(np.ceil((daily_forecast_rate * lead_time) + safety_stock))
            target_stock = int(np.ceil(total_forecast + safety_stock))

            # Reorder Date Calculation
            if daily_forecast_rate > 0:
                days_until_reorder = (target_stock - rop) / daily_forecast_rate
                reorder_date = datetime.now() + timedelta(days=int(days_until_reorder))
                reorder_date_str = reorder_date.strftime('%b %d')
            else:
                reorder_date_str = "N/A"

            velocity = "High" if avg_daily > df[qty_col].mean() else "Normal"
            rop_alert = f"📅 Next Reorder: <b>{reorder_date_str}</b> (at {rop} units)"

            current_stock = int(default_stock)

            report.append({
                'id': str(abs(hash(product))),
                'product': product,
                'avg_daily_sales': round(avg_daily, 1),
                'forecast_30d': int(round(total_forecast)),
                'safety_stock': safety_stock,
                'target_stock': target_stock,
                'rop': rop,
                'rop_alert': rop_alert,
                'velocity': velocity,
                'current_stock_default': current_stock,
                'history_dates': history_dates,
                'history_values': history_values,
                'forecast_dates': future_dates,
                'forecast_values': [round(x, 1) for x in daily_predictions]
            })

        # Group by Category
        grouped_report = {}
        for item in report:
            cat = assign_category(item['product'])
            if cat not in grouped_report: grouped_report[cat] = []
            grouped_report[cat].append(item)

        # Sort Categories
        for cat in grouped_report:
            grouped_report[cat].sort(key=lambda x: x['target_stock'], reverse=True)

        cat_order = ['Accessories', 'Devices', 'Electronics', 'Charging', 'Peripherals', 'Furniture', 'Lifestyle', 'Other']
        sorted_grouped_report = {k: grouped_report[k] for k in cat_order if k in grouped_report}
        for k in grouped_report:
            if k not in sorted_grouped_report: sorted_grouped_report[k] = grouped_report[k]

        return sorted_grouped_report, None

    except Exception as e:
        return None, str(e)