from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # Data
    excel_path: str = "caffe_change.xlsx"
    date_col: str = "Date"
    week_col: str = "Week number"

    # Aggregation
    agg_map: dict = field(default_factory=lambda: {
        "Quantity": "sum",
        "Total": "sum",
        "Discount": "sum",
    })
    rename_map: dict = field(default_factory=lambda: {
        "Quantity": "Total_Units",
        "Total": "Total_Sales",
    })

    # Features
    use_holidays: bool = True
    holiday_country_code: str = "IN"  
    use_discount: bool = True
    use_lags: bool = True

    lag_weeks: List[int] = field(default_factory=lambda: [2, 3])
    fillna_method: str = "mean"  

    # Train/test split
    train_rows: int = 30  # first N rows train, rest test

    # Linear-specific
    plot_coefficients: bool = True
    coef_top_n: int = 20

    # Plot
    plot_xlim: tuple = (30, 52)
    plot_xtick_start: int = 30
    plot_xtick_end: int = 52

   