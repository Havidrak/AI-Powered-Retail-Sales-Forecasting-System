import pandas as pd
import numpy as np
import re
import joblib
import os

# --- 1. HELPER FUNCTIONS ---

def rename_columns_only(df):
    """Smart column renaming to handle messy Excel files."""
    clean_cols = df.columns.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    col_map_ref = dict(zip(clean_cols, df.columns))
    
    targets = {
        'customer_id': ['customerid', 'custid', 'clientid', 'userid', 'id', 'account'],
        'date': ['date', 'time', 'purchasedate', 'timestamp', 'invoice'],
        'total': ['total', 'amount', 'price', 'sales', 'revenue', 'cost'],
        'quantity': ['qty', 'quantity', 'units', 'pieces'],
        'provided_membership': ['member', 'loyalty', 'status', 'type', 'tier']
    }
    
    new_names = {}
    for std, kws in targets.items():
        if std in df.columns: continue
        for clean_col in clean_cols:
            if any(k in clean_col for k in kws):
                new_names[col_map_ref[clean_col]] = std
                break
    
    if new_names:
        df = df.rename(columns=new_names)
    return df

def smart_date_parser(date_series):
    """
    Scans the data to detect if it is DD/MM (Malaysia/UK) or MM/DD (US).
    """
    try:
        sample = date_series.dropna().astype(str).head(100)
        first_nums = []
        second_nums = []
        
        for date_str in sample:
            match = re.search(r'(\d+)[/\-\.](\d+)', date_str)
            if match:
                first_nums.append(int(match.group(1)))
                second_nums.append(int(match.group(2)))
        
        max_first = max(first_nums) if first_nums else 0
        max_second = max(second_nums) if second_nums else 0
        
        if max_first > 12: return True # Day First
        elif max_second > 12: return False # Month First
        else: return True # Default to Malaysia/UK standard
            
    except Exception as e:
        return None # Let pandas guess

# --- 2. DATA PROCESSING ---

def process_company_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.concat(pd.read_excel(file_path, sheet_name=None).values(), ignore_index=True)
            
        df = rename_columns_only(df)
        
        # Basic Cleaning
        if 'customer_id' not in df.columns:
            df['customer_id'] = ['Guest_' + str(i) for i in range(len(df))]
        else:
            df['customer_id'] = df['customer_id'].fillna('Guest_Unknown')

        if 'total' not in df.columns:
            if 'price' in df.columns and 'quantity' in df.columns:
                 df['total'] = pd.to_numeric(df['price'], errors='coerce') * pd.to_numeric(df['quantity'], errors='coerce')
            else:
                 df['total'] = 0
        
        if df['total'].dtype == object:
            df['total'] = df['total'].astype(str).str.replace(r'[$,]', '', regex=True)
        df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0)

        # Smart Date Handling
        if 'date' not in df.columns: df['date'] = pd.Timestamp.now()
        is_day_first = smart_date_parser(df['date'])
        df['date'] = pd.to_datetime(df['date'], dayfirst=is_day_first, errors='coerce').dropna()
        
        # Feature Engineering
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['is_cny'] = df['month'].isin([1, 2]).astype(int)
        df['is_raya'] = df['month'].isin([4, 5, 6]).astype(int)
        df['is_merdeka'] = df['month'].isin([8, 9]).astype(int)
        df['is_deepavali'] = df['month'].isin([10, 11]).astype(int)
        df['is_christmas'] = df['month'].isin([11, 12]).astype(int)

        # Aggregation per Customer
        agg_rules = {
            'total': 'sum',
            'date': ['count', 'min', 'max'],
            'is_weekend': 'mean',
            'is_cny': 'mean', 
            'is_raya': 'mean',
            'is_merdeka': 'mean',
            'is_deepavali': 'mean',
            'is_christmas': 'mean'
        }
        
        has_mem = 'provided_membership' in df.columns
        if has_mem:
            agg_rules['provided_membership'] = lambda x: x.mode()[0] if not x.mode().empty else 'Non-Member'

        summary = df.groupby('customer_id').agg(agg_rules).reset_index()
        
        new_cols = ['customer_id', 'total_spent', 'purchase_count', 'first_purchase', 'last_purchase', 'weekend_ratio', 
                    'is_cny', 'is_raya', 'is_merdeka', 'is_deepavali', 'is_christmas']
        
        if has_mem: new_cols.append('raw_membership')
        summary.columns = new_cols

        # Advanced Metrics (Churn, etc.)
        ref_date = summary['last_purchase'].max()
        summary['recency_days'] = (ref_date - summary['last_purchase']).dt.days
        
        conditions = [
            (summary['recency_days'] <= 30),
            (summary['recency_days'] > 30) & (summary['recency_days'] <= 90),
            (summary['recency_days'] > 90)
        ]
        summary['status'] = np.select(conditions, ['Active', 'Dormant', 'Lost'], default='Lost')

        return summary

    except Exception as e:
        print(f"Processing Error: {e}")
        return pd.DataFrame()

# --- 3. NAMING LOGIC ---

def generate_smart_names(df):
    """
    Analyzes cluster centers to assign descriptive names like:
    'CNY Shoppers (Weekend)', 'CNY Shoppers (VIP)', etc.
    """
    names = {}
    
    seasonal_cols = ['is_cny', 'is_raya', 'is_merdeka', 'is_deepavali', 'is_christmas']
    metrics = ['total_spent', 'weekend_ratio', 'purchase_count'] + seasonal_cols
    existing_metrics = [c for c in metrics if c in df.columns]
    
    means = df.groupby('cluster')[existing_metrics].mean()
    overall_spend_avg = df['total_spent'].mean()
    
    for i in means.index:
        row = means.loc[i]
        label = "Regular Shoppers"
        is_seasonal = False
        
        best_season = None
        highest_score = 0
        
        for season in seasonal_cols:
            if season in row and row[season] > 0.35: 
                if row[season] > highest_score:
                    highest_score = row[season]
                    best_season = season
        
        if best_season:
            is_seasonal = True
            labels_map = {
                'is_cny': 'CNY Crowd',
                'is_raya': 'Raya Shoppers',
                'is_merdeka': 'Merdeka Sale',
                'is_deepavali': 'Deepavali Crowd',
                'is_christmas': 'Year-End Promo'
            }
            base_label = labels_map.get(best_season, "Seasonal")
            
            if row['total_spent'] > (overall_spend_avg * 1.2): suffix = " (High Spenders)"
            elif row['weekend_ratio'] > 0.6: suffix = " (Weekend)"
            elif row['purchase_count'] > 2: suffix = " (Loyal)"
            else: suffix = " (Regular)"
            
            label = base_label + suffix

        if not is_seasonal:
            if row['total_spent'] > df['total_spent'].quantile(0.8): label = "Big Spenders (VIP)"
            elif row['weekend_ratio'] > 0.6: label = "Weekend Warriors"
            elif row['weekend_ratio'] < 0.4: label = "Weekday Regulars"
            else: label = "Occasional Buyers"
            
        names[i] = f"{label}"
        
    return names

# --- 4. MAIN DASHBOARD GENERATOR ---

def generate_dashboard_data(company_csv_path):
    try:
        # Load Model (Graceful Fallback)
        try:
            kmeans = joblib.load(os.path.join(os.path.dirname(__file__), 'kmeans_model_5clusters.joblib'))
            scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler_5clusters.joblib'))
            model_loaded = True
        except:
            model_loaded = False
            
        summary = process_company_data(company_csv_path)
        if summary.empty: return {"error": "Data empty or processing failed"}

        # Prepare Features
        features = ['weekend_ratio', 'is_cny', 'is_raya', 'is_merdeka', 'is_deepavali', 'is_christmas']
        for f in features:
            if f not in summary.columns: summary[f] = 0
            
        # Clustering Execution
        if model_loaded:
            X = summary[features].fillna(0)
            X_scaled = scaler.transform(X)
            summary['cluster'] = kmeans.predict(X_scaled)
            cluster_names = generate_smart_names(summary)
        else:
            # Fallback (Simple Quantile)
            summary['cluster'] = pd.qcut(summary['total_spent'], 5, labels=False, duplicates='drop')
            cluster_names = {0:'Budget', 1:'Regular', 2:'Bronze', 3:'Silver', 4:'Gold'}
            
        summary['cluster_name'] = summary['cluster'].map(cluster_names)

        # Generate Metrics
        returning = len(summary[summary['purchase_count'] > 1])
        new_cust = len(summary) - returning
        
        churn_counts = summary['status'].value_counts().reindex(['Active', 'Dormant', 'Lost']).fillna(0)
        
        qt = summary['total_spent'].quantile([0.33, 0.66])
        low = len(summary[summary['total_spent'] <= qt[0.33]])
        mid = len(summary[(summary['total_spent'] > qt[0.33]) & (summary['total_spent'] <= qt[0.66])])
        high = len(summary[summary['total_spent'] > qt[0.66]])
        
        weekend = len(summary[summary['weekend_ratio'] > 0.5])
        weekday = len(summary) - weekend

        table_stats = summary.groupby('cluster_name').agg({
            'total_spent': 'mean',
            'purchase_count': 'mean',
            'customer_id': 'count'
        }).reset_index()
        table_stats['total_spent'] = table_stats['total_spent'].round(2)
        table_stats['purchase_count'] = table_stats['purchase_count'].round(1)
        
        cluster_counts = summary['cluster_name'].value_counts()
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        bar_data = summary.groupby('cluster_name')['total_spent'].mean().reset_index()
        
        # Final JSON Output
        return {
            'pie_chart': {'labels': cluster_counts.index.tolist(), 'data': cluster_counts.values.tolist(), 'colors': colors},
            'bar_chart': {'labels': bar_data['cluster_name'].tolist(), 'data': bar_data['total_spent'].tolist()},
            'analytics': {
                'retention': [new_cust, returning],
                'churn': churn_counts.values.tolist(),
                'tiers': [low, mid, high],
                'days': [weekday, weekend],
            },
            'table_data': table_stats.to_dict('records'),
            'summary': {
                'total_customers': len(summary),
                'total_members': len(summary[summary['purchase_count']>1])
            }
        }
        
    except Exception as e:
        print(f"Error in dashboard generation: {e}")
        return {"error": str(e)}