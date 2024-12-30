import pytest
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from src.promo_sales_insights.pipelines.model_training.nodes import (
    sort_by_date,
    create_lag_feature,
    create_rolling_mean_feature,
    dropna,
    convert_column_to_datetime,
    encode_categorical,
    split_data,
    apply_winsorize,
    prepare_training_data,
    train_model,
    evaluate_model,
)


def test_sort_by_date():
    df = pd.DataFrame({"date": ["2023-01-02", "2023-01-01"], "value": [10, 20]})
    df = convert_column_to_datetime(df, "date")
    sorted_df = sort_by_date(df, "date")
    assert sorted_df["date"].is_monotonic_increasing


def test_create_lag_feature():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    lagged_df = create_lag_feature(df, "value", 2)
    assert "lag_1" in lagged_df.columns
    assert "lag_2" in lagged_df.columns
    assert lagged_df["lag_1"].iloc[1] == 1


def test_create_rolling_mean_feature():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    rolling_df = create_rolling_mean_feature(df, "value", 2)
    assert "rolling_mean_2" in rolling_df.columns
    assert np.isnan(rolling_df["rolling_mean_2"].iloc[0])


def test_dropna():
    df = pd.DataFrame({"value": [1, np.nan, 3]})
    clean_df = dropna(df)
    assert clean_df.isna().sum().sum() == 0


def test_convert_column_to_datetime():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
    df = convert_column_to_datetime(df, "date")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_encode_categorical():
    df = pd.DataFrame({"category": ["A", "B", "A"]})
    encoded_df = encode_categorical(df)
    assert pd.api.types.is_integer_dtype(encoded_df["category"])


def test_split_data():
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=5),
            "value": [1, 2, 3, 4, 5],
        }
    )
    train, test = split_data(df, "2023-01-03", "2023-01-05", "date")
    assert len(train) == 3
    assert len(test) == 2


def test_apply_winsorize():
    df = pd.DataFrame({"value": [1, 2, 100]})
    winsorized_df = apply_winsorize(df, "value", "winsorized", 0.1, 0.9)
    assert winsorized_df["winsorized"].max() <= df["value"].quantile(0.9)


def test_prepare_training_data():
    df = pd.DataFrame(
        {"date": ["2023-01-01", "2023-01-02"], "target": [1, 2], "feature": [3, 4]}
    )
    train, test = split_data(df, "2023-01-01", "2023-01-02", "date")
    X_train, y_train, X_test, y_test = prepare_training_data(
        train, test, "target", "date"
    )
    assert X_train.shape[1] == 1  # Only "feature" column remains


def test_train_model():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    y_train = pd.Series([1, 2, 3])
    X_test = pd.DataFrame({"feature": [4, 5]})
    y_test = pd.Series([4, 5])
    params = {"n_estimators": 10}
    model, comparison_df = train_model(
        X_train, y_train, X_test, y_test, params, 42, "target", "prediction"
    )
    assert isinstance(model, LGBMRegressor)
    assert "prediction" in comparison_df.columns


def test_evaluate_model():
    comparison_df = pd.DataFrame({"target": [1, 2], "prediction": [1.1, 1.9]})
    rmse, mae = evaluate_model(comparison_df, "target", "prediction")
    assert rmse > 0
    assert mae > 0
