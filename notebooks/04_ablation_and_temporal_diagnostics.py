import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ensure we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import walk_forward_cv, summarise_cv, get_feature_cols

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

def run_diagnostics(df):
    print("Running Temporal Diagnostics...")
    # Plot ACF/PACF for orders, tickets, footfall, rain
    targets = ["orders_count", "food_tickets_count", "suffolk_nassau_footfall", "rain_mm"]
    
    fig, axes = plt.subplots(len(targets), 2, figsize=(15, 3.5 * len(targets)))
    
    for i, col in enumerate(targets):
        if col in df.columns:
            ts = df[col].dropna()
            if len(ts) > 0:
                plot_acf(ts, ax=axes[i, 0], lags=168, title=f"ACF: {col}")
                plot_pacf(ts, ax=axes[i, 1], lags=48, title=f"PACF: {col}", method='ywm')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "acf_pacf_diagnostics.png")
    plt.close()
    
    # Cross correlation: footfall vs orders
    if "suffolk_nassau_footfall" in df.columns and "orders_count" in df.columns:
        valid = df[["orders_count", "suffolk_nassau_footfall"]].dropna()
        if not valid.empty:
            lags = range(-24, 25)
            xcorr = [valid["orders_count"].corr(valid["suffolk_nassau_footfall"].shift(l)) for l in lags]
            plt.figure(figsize=(10, 4))
            plt.bar(lags, xcorr)
            plt.title("Cross-Correlation: Footfall (lagged) vs Orders")
            plt.xlabel("Lag (hours) - negative means Footfall leads")
            plt.ylabel("Correlation")
            plt.savefig(OUTPUTS_DIR / "crosscorr_footfall_orders.png")
            plt.close()

def run_ablation(df):
    print("\nRunning Ablation Matrix...")
    target = "orders_count"
    
    temporal_cols = [
        "orders_count_lag_24h", "orders_count_lag_48h", "orders_count_lag_168h", "orders_count_lag_336h",
        "food_tickets_count_lag_24h", "food_tickets_count_lag_48h", "food_tickets_count_lag_168h", "food_tickets_count_lag_336h",
        "orders_count_roll_mean_24", "orders_count_roll_mean_168", "orders_count_roll_std_24",
        "food_tickets_count_roll_mean_24", "food_tickets_count_roll_mean_168",
        "hour", "weekday", "month", "quarter", "is_weekend", "is_friday_saturday", "week_of_year",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",
        "is_food_service", "is_breakfast_window", "is_lunch_window", "is_dinner_window", "is_after_food_close", "hours_since_food_close",
        "is_live_music_window", "weekend_x_music"
    ]
    weather_cols = ["temp_c", "rain_mm", "wind_speed_kmh", "weather_severity_flag", "rain_x_weekend"]
    event_cols = [
        "major_sports_event_flag", "city_event_flag", "st_patricks_week_flag", "special_event_flag", "event_intensity",
        "aviva_event_flag", "croke_park_event_flag", "nearby_venue_event_flag", "event_impact_score",
        "bloomsday_flag", "bloomsday_week_flag", "summer_tourism_flag", "christmas_market_flag", "new_years_eve_flag", "new_years_day_flag",
        "bank_holiday_flag", "school_holiday_flag"
    ]
    footfall_cols = ["suffolk_footfall_lag_24h", "suffolk_footfall_lag_168h", "suffolk_footfall_roll_24h"]
    airport_cruise_cols = ["airport_arrivals", "airport_arrivals_lag1", "airport_arrivals_zscore", "cruise_ship_flag", "ships_in_port_count", "cruise_passenger_estimate"]
    failte_cols = ["failte_event_count", "failte_free_event_count", "failte_festival_count", "college_term_flag", "days_from_payday"]
    
    ablations = {
        "A1: Temporal": temporal_cols,
        "A2: +Weather": temporal_cols + weather_cols,
        "A3: +Events": temporal_cols + weather_cols + event_cols,
        "A4: +Footfall": temporal_cols + weather_cols + event_cols + footfall_cols,
        "A5: +Airport/Cruise": temporal_cols + weather_cols + event_cols + footfall_cols + airport_cruise_cols,
        "A6: +Failte/Proxies": temporal_cols + weather_cols + event_cols + footfall_cols + airport_cruise_cols + failte_cols,
    }
    
    results = []
    baseline_mae_mean = None
    
    for name, cols in ablations.items():
        print(f"Evaluating {name} ({len(cols)} features)...")
        features = get_feature_cols(df, cols)
        b_res, x_res = walk_forward_cv(df, target, features, n_splits=5, test_weeks=4, min_train_weeks=12, verbose=False)
        b_sum = summarise_cv(b_res, "Baseline")
        x_sum = summarise_cv(x_res, "XGBoost")
        
        if baseline_mae_mean is None:
            baseline_mae_mean = b_sum["mae"].mean()
            results.append({
                "Run": "A0: Baseline (Last Week)", 
                "Features": 0, 
                "MAE": baseline_mae_mean, 
                "Shift Accuracy": b_sum["shift_accuracy"].mean()
            })
            
        results.append({
            "Run": name,
            "Features": len(features),
            "MAE": x_sum["mae"].mean(),
            "Shift Accuracy": x_sum["shift_accuracy"].mean()
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUTS_DIR / "ablation_results.csv", index=False)
    
    with open(OUTPUTS_DIR / "ablation_summary.md", "w") as f:
        f.write("# Ablation Matrix Results\n\n")
        f.write(res_df.to_markdown(index=False))
        
    print("\nAblation complete! Results saved to outputs/")

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/features.parquet")
    run_diagnostics(df)
    run_ablation(df)
