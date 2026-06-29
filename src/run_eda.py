import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Output directory within the project workspace
OUT_DIR = "notebooks/eda"
os.makedirs(OUT_DIR, exist_ok=True)

# Set styling
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#0A0A0A", "grid.color": "#27272A"})

def assign_shift(hour):
    if 12 <= hour <= 15: return 'Lunch'
    if 17 <= hour <= 21: return 'Evening'
    if hour >= 21 or hour <= 2: return 'Late Bar'
    return 'Off-Peak'

try:
    print("Loading feature matrix...")
    df = pd.read_parquet("data/processed/features.parquet")
    df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])
    df['hour'] = df['timestamp_hour'].dt.hour
    df['day_of_week'] = df['timestamp_hour'].dt.day_name()
    df['shift'] = df['hour'].apply(assign_shift)

    # 1. Pearson Correlation Table
    print("Calculating correlations...")
    # Select key numerical features
    cols_to_correlate = [
        'orders_count', 'suffolk_footfall_lag_24h', 'suffolk_footfall_roll_24h', 
        'rain_mm', 'temp_c', 'event_intensity', 'failte_event_count', 'airport_arrivals_zscore'
    ]
    # Keep only columns that exist
    cols_to_correlate = [c for c in cols_to_correlate if c in df.columns]
    
    corr_matrix = df[cols_to_correlate].corr(method='pearson')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0, vmin=-1, vmax=1, 
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Pearson Correlation: Signals vs. Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_correlation_matrix.png"), dpi=150)
    plt.close()

    # Save numeric correlation specifically against orders_count
    target_corr = corr_matrix[['orders_count']].sort_values(by='orders_count', ascending=False)
    target_corr.to_csv(os.path.join(OUT_DIR, "01_orders_correlation.csv"))

    # 2. Footfall vs Orders by Shift
    if 'suffolk_footfall_roll_24h' in df.columns:
        print("Plotting Footfall vs Demand...")
        plt.figure(figsize=(10, 6))
        # Filter out Off-Peak for clearer visualization
        shift_df = df[df['shift'] != 'Off-Peak']
        sns.scatterplot(data=shift_df, x='suffolk_footfall_roll_24h', y='orders_count', 
                        hue='shift', alpha=0.5, palette=['#3B82F6', '#F59E0B', '#10B981'])
        plt.title("Footfall vs. Orders (By Shift)")
        plt.xlabel("Suffolk St Footfall (24h Rolling Avg)")
        plt.ylabel("Orders Count")
        plt.legend(title='Shift')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "02_footfall_vs_orders.png"), dpi=150)
        plt.close()

    # 3. Anomaly Cases (St. Patrick's Week & Aviva Events)
    print("Plotting Anomaly Cases...")
    plt.figure(figsize=(10, 6))
    
    # Create an anomaly flag
    df['anomaly_type'] = 'Normal Day'
    if 'aviva_event_flag' in df.columns:
        df.loc[df['aviva_event_flag'] == 1, 'anomaly_type'] = 'Aviva Event'
    if 'st_patricks_week_flag' in df.columns:
        df.loc[df['st_patricks_week_flag'] == 1, 'anomaly_type'] = "St. Patrick's Week"
        
    sns.boxplot(data=df, x='anomaly_type', y='orders_count', order=['Normal Day', 'Aviva Event', "St. Patrick's Week"], palette='muted')
    plt.title("Demand Distribution: Normal vs. Anomaly Events")
    plt.ylabel("Orders Count")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_anomaly_events.png"), dpi=150)
    plt.close()

    # 4. Residual Analysis
    print("Generating Residual Analysis...")
    cv_preds_path = "models/cv_out_of_sample_preds_orders_count.csv"
    if os.path.exists(cv_preds_path):
        cv_df = pd.read_csv(cv_preds_path)
        cv_df['timestamp_hour'] = pd.to_datetime(cv_df['timestamp_hour'])
        
        # Merge with main df to get day_of_week and shift
        cv_df = pd.merge(cv_df, df[['timestamp_hour', 'day_of_week', 'shift']], on='timestamp_hour', how='inner')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=cv_df, x='day_of_week', y='residual', hue='shift',
                    order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    palette='dark')
        plt.axhline(0, color='white', linestyle='--', alpha=0.5)
        plt.title("XGBoost Out-of-Sample Residuals by Day and Shift")
        plt.ylabel("Residual Error")
        plt.xlabel("Day of Week")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "04_residual_analysis.png"), dpi=150)
        plt.close()
    else:
        print("Out-of-sample CV predictions not found. Run model.py first.")

    print(f"EDA successfully completed. Artifacts saved to {OUT_DIR}/")
    
except Exception as e:
    print(f"Error generating EDA: {e}")
