import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgboost
from xgboost import plot_tree

import holidays as hdays

from project_config import Config

def add_holiday_flag(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.use_holidays:
        return df

    try:
        holiday_cls = getattr(hdays, cfg.holiday_country_code)
    except AttributeError as e:
        raise ValueError(
            f"holiday_country_code='{cfg.holiday_country_code}' not found in holidays library."
        ) from e

    holidays_obj = holiday_cls()
    df["Is_holiday"] = df[cfg.date_col].apply(lambda x: 1 if x in holidays_obj else 0)
    return df

def make_weekly(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    weekly = (
        df.groupby([cfg.week_col])
          .agg(cfg.agg_map)
          .reset_index()
          .rename(columns=cfg.rename_map)
    )

    weekly.sort_values(by=[cfg.week_col], inplace=True)

    if not cfg.use_discount and "Discount" in weekly.columns:
        weekly = weekly.drop(columns=["Discount"])

    # Lag features
    if cfg.use_lags:
        for k in cfg.lag_weeks:
            weekly[f"Lag_{k}_Units"] = weekly["Total_Units"].shift(k)

    # Fill NA
    if cfg.fillna_method == "mean":
        weekly = weekly.fillna(weekly.mean(numeric_only=True))
    elif cfg.fillna_method == "zero":
        weekly = weekly.fillna(0)
    else:
        raise ValueError("fillna_method must be 'mean' or 'zero'")

    return weekly


def train_test_split_time(weekly: pd.DataFrame, cfg: Config):
    X_data = weekly.drop(columns="Total_Units")
    y_data = weekly["Total_Units"]

    x_train = X_data.iloc[: cfg.train_rows].copy()
    x_test = X_data.iloc[cfg.train_rows :].copy()
    y_train = y_data.iloc[: cfg.train_rows].copy()
    y_test = y_data.iloc[cfg.train_rows :].copy()

    return x_train, x_test, y_train, y_test


def fit_xgb(x_train, y_train, cfg: Config):
    model = xgboost.XGBRegressor(**cfg.xgb_params)
    model.fit(x_train, y_train)
    return model


def print_metrics(y_true, y_pred):
    print("XGBoost Model Performance:")
    print("R^2 Score:", r2_score(y_true, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_true, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_true, y_pred))


def print_feature_map(weekly: pd.DataFrame):
    feature_names = weekly.drop(columns=["Total_Units"]).columns
    print("\nFeature Names:")
    for i, n in enumerate(feature_names):
        print(f"f{i} = {n}")

def plot_predictions(weekly: pd.DataFrame, x_test: pd.DataFrame, y_pred: np.ndarray, cfg: Config):
    week_nums_test = x_test[cfg.week_col].values

    # sort by week number so the line looks correct over time
    order = week_nums_test.argsort()
    week_nums_test_sorted = week_nums_test[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(10, 5))
    plt.plot(weekly[cfg.week_col], weekly["Total_Units"], label="Actual Total Units")
    plt.plot(week_nums_test_sorted, y_pred_sorted, "o-", label="XGBoost Predictions (test)")
    plt.xlabel(cfg.week_col)
    plt.ylabel("Total Units")
    plt.legend()
    plt.title("XGBoost Model Predictions vs Actual Total Units")
    plt.grid(True, alpha=0.3)

    plt.xlim(*cfg.plot_xlim)
    plt.xticks(np.arange(cfg.plot_xtick_start, cfg.plot_xtick_end + 1, 1))
    plt.show()


def main():
    cfg = Config()

    df = pd.read_excel(cfg.excel_path)
    df = add_holiday_flag(df, cfg)

    weekly = make_weekly(df, cfg)

    x_train, x_test, y_train, y_test = train_test_split_time(weekly, cfg)

    model = fit_xgb(x_train, y_train, cfg)
    y_pred = model.predict(x_test)

    print_metrics(y_test, y_pred)
    print_feature_map(weekly)

    print("\nBooster feature importance (raw score dict):")
    print(model.get_booster().get_score())
    plot_predictions(weekly, x_test, y_pred, cfg)


if __name__ == "__main__":
    main()