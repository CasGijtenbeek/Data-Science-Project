import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import holidays as hdays

from linear_config import Config

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

def plot_predictions(weekly: pd.DataFrame, x_test: pd.DataFrame, y_pred: np.ndarray, cfg: Config):
    week_nums_test = x_test[cfg.week_col].values

    order = week_nums_test.argsort()
    week_nums_test_sorted = week_nums_test[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(10, 5))
    plt.plot(weekly[cfg.week_col], weekly["Total_Units"], label="Actual Total Units")
    plt.plot(week_nums_test_sorted, y_pred_sorted, "o-", label="Linear Regression Predictions (test)")
    plt.xlabel(cfg.week_col)
    plt.ylabel("Total Units")
    plt.legend()
    plt.title("Linear Regression Predictions vs Actual Total Units")
    plt.grid(True, alpha=0.3)

    plt.xlim(*cfg.plot_xlim)
    plt.xticks(np.arange(cfg.plot_xtick_start, cfg.plot_xtick_end + 1, 1))
    plt.show()

def fit_linear(x_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    model.fit(x_train, y_train)
    return model

def get_linear_coefficients(model, feature_names, cfg: Config):
    if not cfg.plot_coefficients:
        return None

    lr = model.named_steps["lr"]  # grab LinearRegression inside pipeline

    coefs = pd.Series(lr.coef_, index=feature_names)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)


    return coefs


def main():
    cfg = Config()

    df = pd.read_excel(cfg.excel_path)
    df = add_holiday_flag(df, cfg)
    weekly = make_weekly(df, cfg)

    print("\nActive features:")
    print(weekly.drop(columns="Total_Units").columns.tolist())

    x_train, x_test, y_train, y_test = train_test_split_time(weekly, cfg)

    model = fit_linear(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Linear Regression Performance:")
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    coefs = get_linear_coefficients(model, x_train.columns, cfg)

    lr = model.named_steps["lr"]
    print("\nIntercept:", lr.intercept_)

    if coefs is not None:
        print("\nCoefficients (sorted by abs):")
        print(coefs)

    plot_predictions(weekly, x_test, y_pred, cfg)


if __name__ == "__main__":
    main()


