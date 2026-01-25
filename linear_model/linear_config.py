from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
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
    # lag_2_name: str = "Lag_2"
    # roll3_name: str = "Rolling_Avg_3"
    # rolling_window: int = 3
    
    # Train/test split
    train_frac: float = 0.8

    # Recursive forecasting
    warmup_n_dates: int = 3

    # Linear model options
    standardize: bool = True

    # Plot
    plot_figsize: Tuple[int, int] = (12, 5)


   