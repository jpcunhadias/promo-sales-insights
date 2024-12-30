"""
This is a boilerplate test file for pipeline 'model_training'
generated using Kedro 0.19.10.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
import pytest
from lightgbm import LGBMRegressor
from src.promo_sales_insights.pipelines.model_training.nodes import (
    convert_column_to_str, sort_by_date, create_lag_feature,
    create_rolling_mean_feature, dropna, convert_column_to_datetime,
    encode_categorical, split_data, apply_winsorize, train_model, evaluate_model
)

def test_convert_column_to_str():
    df = pd.DataFrame({'A': [1, 2, 3]})
    result = convert_column_to_str(df, 'A')
    assert result['A'].dtype == object
    assert result['A'].iloc[0] == '1'

def test_sort_by_date():
    df = pd.DataFrame({'date': ['2023-01-03', '2023-01-01', '2023-01-02']})
    df = convert_column_to_datetime(df, 'date')
    result = sort_by_date(df, 'date')
    assert result['date'].iloc[0] == pd.Timestamp('2023-01-01')

def test_create_lag_feature():
    df = pd.DataFrame({'target': [10, 20, 30, 40, 50]})
    result = create_lag_feature(df, 'target', lags=2)
    assert 'lag_1' in result.columns
    assert 'lag_2' in result.columns
    assert result['lag_1'].iloc[2] == 20

def test_create_rolling_mean_feature():
    df = pd.DataFrame({'target': [10, 20, 30, 40, 50]})
    result = create_rolling_mean_feature(df, 'target', window=3)
    assert 'rolling_mean_3' in result.columns
    assert result['rolling_mean_3'].iloc[4] == np.mean([30, 40, 50])

def test_dropna():
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
    result = dropna(df)
    assert result.isnull().sum().sum() == 0
    assert len(result) == 1

def test_convert_column_to_datetime():
    df = pd.DataFrame({'date': ['2023-01-01', '2023-02-01']})
    result = convert_column_to_datetime(df, 'date')
    assert result['date'].dtype == 'datetime64[ns]'

def test_encode_categorical():
    df = pd.DataFrame({'A': ['cat', 'dog', 'cat']})
    result = encode_categorical(df)
    assert result['A'].dtype == 'int64'  # Adjust to match the actual output
    assert result['A'].nunique() == 2

def test_split_data():
    df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=6),
                       'value': [10, 20, 30, 40, 50, 60]})
    train, test = split_data(df, '2023-01-04', '2023-01-06', 'date')
    assert len(train) == 4
    assert len(test) == 2

def test_apply_winsorize():
    df = pd.DataFrame({'A': [1, 2, 3, 100, 200]})
    result = apply_winsorize(df, 'A', 'A_winsorized', 0.1, 0.9)
    assert result['A_winsorized'].iloc[-1] == result['A'].quantile(0.9)

def test_train_model():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40],
        'target': [100, 200, 300, 400],
        'date': pd.date_range('2023-01-01', periods=4)
    })
    train, test = split_data(df, '2023-01-03', '2023-01-04', 'date')
    best_params = {'n_estimators': 10, 'learning_rate': 0.1, 'max_depth': 3}
    model, test_result = train_model(
        train, test, best_params, random_state=42,
        target_column='target', datetime_column='date', prediction_column='predictions'
    )
    assert isinstance(model, LGBMRegressor)
    assert 'predictions' in test_result.columns

def test_evaluate_model():
    df = pd.DataFrame({'actual': [100, 200, 300], 'predicted': [110, 190, 310]})
    rmse, mae = evaluate_model(df, 'actual', 'predicted')
    assert np.isclose(rmse, 10.0, atol=1e-2)
    assert np.isclose(mae, 10.0, atol=1e-2)