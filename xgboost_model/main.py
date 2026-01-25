import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgboost


from project_config import Config



def encode_weekday(d: pd.Series, cfg: Config) -> pd.Series:
    cat = pd.Categorical(d, categories=list(cfg.week_order), ordered=True)
    return cat.codes

def series_key(df: pd.DataFrame, cfg: Config):
    dummy_cols = [c for c in df.columns if c.startswith("Category_")]
    return [cfg.bucket_col] + dummy_cols

def run_normal_xgb(x_train, y_train, x_test, y_test, cfg: Config):
    model = xgboost.XGBRegressor(**cfg.xgb_params)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    daily_sum = evaluate_daily_totals(x_test, y_test, preds)
    plot_daily_sum(daily_sum, cfg)


def run_recursive_xgb(x_train, y_train, x_test, y_test, cfg: Config):
    xgb_bootstrap = train_bootstrap(x_train, y_train, cfg)
    x_train_rec = build_recursive_training_features(x_train, xgb_bootstrap, cfg)

    xgb_rec = xgboost.XGBRegressor(**cfg.xgb_params)
    xgb_rec.fit(x_train_rec, y_train)

    preds = recursive_forecast_groupwise(x_test, y_test, xgb_rec, cfg)

    daily_sum = evaluate_daily_totals(x_test, y_test, preds)
    plot_daily_sum(daily_sum, cfg)



def build_daily_df(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])

    # 1) Aggregate at Date + Time Bucket + Category
    daily_df = (
        df.groupby([cfg.date_col, cfg.bucket_col, cfg.category_col])
          .agg({
              cfg.qty_col: "sum",
              cfg.weekday_col: "first",
          })
          .reset_index()
          .rename(columns={cfg.qty_col: "Total_Units"})
    )

    daily_df = daily_df.sort_values(by=[cfg.date_col, cfg.bucket_col, cfg.category_col]).reset_index(drop=True)

    # 2) Encode weekday 
    daily_df[cfg.weekday_col] = encode_weekday(daily_df[cfg.weekday_col], cfg)

    # 3) One-hot categories 
    cat_dummies = pd.get_dummies(daily_df[cfg.category_col], prefix="Category")
    daily_df = pd.concat([daily_df.drop(columns=[cfg.category_col]), cat_dummies], axis=1)

    # 4) Define group columns for lagging:
    dummy_cols = [c for c in daily_df.columns if c.startswith("Category_")]
    group_cols = [cfg.bucket_col] + dummy_cols

    # 5) Create grouped lags + rolling avg
    daily_df[cfg.lag_1_name] = daily_df.groupby(group_cols)["Total_Units"].shift(1)
    daily_df[cfg.lag_2_name] = daily_df.groupby(group_cols)["Total_Units"].shift(2)
    daily_df[cfg.roll3_name] = (
        daily_df.groupby(group_cols)["Total_Units"]
                .transform(lambda x: x.rolling(window=cfg.rolling_window, min_periods=1).mean())
    )

    # 6) Drop NA rows caused by lags 
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

def train_bootstrap(x_train: pd.DataFrame, y_train: pd.Series, cfg: Config):
    model = xgboost.XGBRegressor(**cfg.xgb_params)
    model.fit(x_train, y_train)
    return model

def build_recursive_training_features(x_train: pd.DataFrame, bootstrap_model, cfg: Config) -> pd.DataFrame:
   
    x_train_rec = x_train.copy()

    lag1_col = x_train_rec.columns.get_loc(cfg.lag_1_name)
    lag2_col = x_train_rec.columns.get_loc(cfg.lag_2_name)
    roll3_col = x_train_rec.columns.get_loc(cfg.roll3_name)

    for i in range(2, len(x_train_rec) - 1):
        X_cur = x_train_rec.iloc[i:i+1]
        y_hat = float(bootstrap_model.predict(X_cur)[0])

        # update next row using prediction
        x_train_rec.iloc[i+1, lag2_col] = x_train_rec.iloc[i, lag1_col]
        x_train_rec.iloc[i+1, lag1_col] = y_hat

        last_3 = [
            y_hat,
            float(x_train_rec.iloc[i, lag1_col]),
            float(x_train_rec.iloc[i, lag2_col]),
        ]
        x_train_rec.iloc[i+1, roll3_col] = float(np.mean(last_3))

    return x_train_rec

def recursive_forecast_groupwise(x_test: pd.DataFrame, y_test: pd.Series, xgb_rec, cfg: Config) -> np.ndarray:
    x_test_rec = x_test.copy()
    preds = pd.Series(index=x_test_rec.index, dtype=float)

    key_cols = series_key(x_test_rec, cfg)

    x_test_rec = x_test_rec.reset_index()  
    y_test_reset = y_test.reset_index(drop=True)

    lag1 = cfg.lag_1_name
    lag2 = cfg.lag_2_name
    roll3 = cfg.roll3_name

    # Warmup dates 
    warmup_dates = x_test_rec[cfg.date_col].drop_duplicates().iloc[: cfg.warmup_n_dates]
    warmup_mask = x_test_rec[cfg.date_col].isin(warmup_dates)

    # Process each series separately
    for _, idxs in x_test_rec.groupby(key_cols).groups.items():
        idxs = list(idxs)
        # Sort within series by Date
        idxs = sorted(idxs, key=lambda i: x_test_rec.loc[i, cfg.date_col])

        for j, i in enumerate(idxs):
            if warmup_mask.iloc[i]:
                y_pred = float(y_test_reset.iloc[i])
            else:
                X_cur = x_test_rec.loc[[i], x_test.columns.tolist()]  
                y_pred = float(xgb_rec.predict(X_cur)[0])

            preds.iloc[i] = y_pred

            
            if j + 1 < len(idxs):
                nxt = idxs[j + 1]
                x_test_rec.loc[nxt, lag2] = x_test_rec.loc[i, lag1]
                x_test_rec.loc[nxt, lag1] = y_pred

                last_3 = [y_pred, float(x_test_rec.loc[i, lag1]), float(x_test_rec.loc[i, lag2])]
                x_test_rec.loc[nxt, roll3] = float(np.mean(last_3))

    return preds.to_numpy()


def evaluate_daily_totals(x_test: pd.DataFrame, y_test: pd.Series, preds: np.ndarray):
    pred_df = pd.DataFrame(
        {"Date": x_test.index, "Actual": y_test.values, "Predicted": preds}
    ).set_index("Date")

    daily_sum = pred_df.groupby("Date").sum()

    print("XGBoost Model Performance (Aggregated Daily):")
    print("R^2 Score:", r2_score(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MAE:", mean_absolute_error(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MSE:", mean_squared_error(daily_sum["Actual"], daily_sum["Predicted"]))
    print("MAPE:", np.mean(np.abs((daily_sum["Actual"] - daily_sum["Predicted"]) / daily_sum["Actual"])) * 100)

    return daily_sum


def plot_daily_sum(daily_sum: pd.DataFrame, cfg: Config):
    plt.figure(figsize=cfg.plot_figsize)
    plt.plot(daily_sum.index, daily_sum["Actual"], label="Actual", marker="o", linestyle="-")
    plt.plot(daily_sum.index, daily_sum["Predicted"], label="Predicted (XGB)", marker="o", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Total Units")
    plt.title("Actual vs Predicted Total Units (Aggregated Daily) - Recursive Forecasting")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    cfg = Config()

    df = pd.read_excel(cfg.excel_path)
    
    daily_df = build_daily_df(df, cfg)

    copy_daily_df = daily_df.copy()

    # Train/test split
    x_train, y_train, x_test, y_test = split_train_test(copy_daily_df, cfg)

    mode = cfg.forecast_mode.lower().strip()

    if mode == "normal":
        run_normal_xgb(x_train, y_train, x_test, y_test, cfg)

    elif mode == "recursive":
        run_recursive_xgb(x_train, y_train, x_test, y_test, cfg)

    else:
        raise ValueError("Config.forecast_mode must be 'normal' or 'recursive'")


if __name__ == "__main__":
    main()