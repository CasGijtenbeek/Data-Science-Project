from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Config:

    #forecasting mode ("normal" or "recursive")
    
    # forecast_mode: str = 'recursive'
    forecast_mode: str = 'normal'

    # Data
    excel_path: str = "caffe_change.xlsx"
    date_col: str = "Date"
    qty_col: str = "Quantity"
    weekday_col: str = "Week day"
    bucket_col: str = "Time Bucket Num"
    category_col: str = "Category"

    week_order: Tuple[str, ...] = (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    )

    # Lags / rolling
    lag_1_name: str = "Lag_1"
    lag_2_name: str = "Lag_2"
    roll3_name: str = "Rolling_Avg_3"
    rolling_window: int = 3

    # Features
    use_weekday: bool = True
    use_weeknum: bool = True
    use_day_of_month: bool = True
    lag_days: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Train/test split
    train_frac: float = 0.8

    
    warmup_n_dates: int = 3            

    # Model hyperparams
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    })

    # Plot
    plot_figsize: Tuple[int, int] = (12, 5)

