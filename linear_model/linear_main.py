import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_config import Config


def encode_weekday(d: pd.Series, cfg: Config) -> pd.Series:
    cat = pd.Categorical(d, categories=list(cfg.week_order), ordered=True)
    return cat.codes

def series_key_from_features(X: pd.DataFrame, cfg: Config):
    dummy_cols = [c for c in X.columns if c.startswith("Category_")]
    return [cfg.bucket_col] + dummy_cols


def build_daily_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df[cfg.category_col] = df[cfg.category_col].astype(str).str.strip()

    daily_df = (
        df.groupby([cfg.date_col, cfg.bucket_col, cfg.category_col])
          .agg({
              cfg.qty_col: "sum",
              cfg.weekday_col: "first",
          })
          .reset_index()
          .rename(columns={cfg.qty_col: "Total_Units"})
    )

    daily_df = daily_df.sort_values(
        by=[cfg.date_col, cfg.bucket_col, cfg.category_col]
    ).reset_index(drop=True)

    # Weekday encoding
    daily_df[cfg.weekday_col] = encode_weekday(daily_df[cfg.weekday_col], cfg)

    cat_dummies = pd.get_dummies(daily_df[cfg.category_col], prefix="Category")
    daily_df = pd.concat(
        [daily_df.drop(columns=[cfg.category_col]), cat_dummies],
        axis=1
    )

    dummy_cols = [c for c in daily_df.columns if c.startswith("Category_")]
    group_cols = [cfg.bucket_col] + dummy_cols

    daily_df[cfg.lag_1_name] = (
        daily_df.groupby(group_cols)["Total_Units"].shift(1)
    )

    daily_df = daily_df.dropna().reset_index(drop=True)
    return daily_df

def split_train_test(daily_df: pd.DataFrame, cfg: Config):
    df_idx = daily_df.copy().set_index(cfg.date_col)

    X_all = df_idx.drop(columns=["Total_Units"])
    y_all = df_idx["Total_Units"]

    split_index = int(len(X_all) * cfg.train_frac)
    x_train = X_all.iloc[:split_index].copy()
    y_train = y_all.iloc[:split_index].copy()
    x_test = X_all.iloc[split_index:].copy()
    y_test = y_all.iloc[split_index:].copy()

    return x_train, y_train, x_test, y_test

def make_linear_model(cfg: Config):
    est = LinearRegression()
    if cfg.standardize:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return Pipeline([("model", est)])


def train_bootstrap(x_train: pd.DataFrame, y_train: pd.Series, cfg: Config):
    model = make_linear_model(cfg)
    model.fit(x_train, y_train)
    return model


def build_recursive_training_features_groupwise(x_train: pd.DataFrame, bootstrap_model, cfg: Config) -> pd.DataFrame:
    
    X = x_train.copy().reset_index()  
    feature_cols = x_train.columns.tolist()
    lag1 = cfg.lag_1_name

    for _, idxs in X.groupby([cfg.bucket_col]).groups.items():
        idxs = sorted(list(idxs), key=lambda i: X.loc[i, cfg.date_col])

        for j in range(1, len(idxs) - 1):
            i = idxs[j]
            nxt = idxs[j + 1]

            X_cur = X.loc[[i], feature_cols]
            y_hat = float(bootstrap_model.predict(X_cur)[0])

            X.loc[nxt, lag1] = y_hat

    return X.set_index(cfg.date_col)[feature_cols]


def recursive_forecast_groupwise(x_test: pd.DataFrame, y_test: pd.Series, final_model, cfg: Config) -> np.ndarray:
    X = x_test.copy().reset_index()
    y = y_test.reset_index(drop=True)

    feature_cols = x_test.columns.tolist()
    lag1 = cfg.lag_1_name

    warmup_dates = X[cfg.date_col].drop_duplicates().iloc[: cfg.warmup_n_dates]
    warmup_mask = X[cfg.date_col].isin(warmup_dates)

    preds = np.zeros(len(X), dtype=float)

    for _, idxs in X.groupby([cfg.bucket_col]).groups.items():
        idxs = sorted(list(idxs), key=lambda i: X.loc[i, cfg.date_col])

        for j, i in enumerate(idxs):
            if warmup_mask.iloc[i]:
                y_pred = float(y.iloc[i])  
            else:
                X_cur = X.loc[[i], feature_cols]
                y_pred = float(final_model.predict(X_cur)[0])

            preds[i] = y_pred

            if j + 1 < len(idxs):
                nxt = idxs[j + 1]
                X.loc[nxt, lag1] = y_pred  

    return preds

def evaluate_daily_totals(x_test: pd.DataFrame, y_test: pd.Series, preds: np.ndarray):
    pred_df = pd.DataFrame(
        {"Date": x_test.index, "Actual": y_test.values, "Predicted": preds}
    ).set_index("Date")

    daily_sum = pred_df.groupby("Date").sum()

    print("Linear Model Performance (Aggregated Daily):")
    print("R^2 Score:", r2_score(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MAE:", mean_absolute_error(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MSE:", mean_squared_error(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MAPE:", np.mean(np.abs((daily_sum["Actual"] - daily_sum["Predicted"]) / daily_sum["Actual"])) * 100)

    return daily_sum


def plot_daily_sum(daily_sum: pd.DataFrame, cfg: Config):
    plt.figure(figsize=cfg.plot_figsize)
    plt.plot(daily_sum.index, daily_sum["Actual"], label="Actual", marker="o", linestyle="-")
    plt.plot(daily_sum.index, daily_sum["Predicted"], label="Predicted (Linear)", marker="o", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Total Units")
    plt.title("Actual vs Predicted Total Units (Aggregated Daily) - Recursive Forecasting (Linear)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def main():
    cfg = Config()

    df = pd.read_excel(cfg.excel_path)
  
    daily_df = build_daily_df(df, cfg)

    x_train, y_train, x_test, y_test = split_train_test(daily_df, cfg)

    bootstrap = train_bootstrap(x_train, y_train, cfg)
    x_train_rec = build_recursive_training_features_groupwise(x_train, bootstrap, cfg)

    final_model = make_linear_model(cfg)
    final_model.fit(x_train_rec, y_train)

    preds = recursive_forecast_groupwise(x_test, y_test, final_model, cfg)

    daily_sum = evaluate_daily_totals(x_test, y_test, preds)
    plot_daily_sum(daily_sum, cfg)


if __name__ == "__main__":
    main()


