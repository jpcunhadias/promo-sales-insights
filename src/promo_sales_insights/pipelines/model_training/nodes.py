"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.10
"""

import logging
from typing import Tuple

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sort_by_date(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Sorts the DataFrame by a specified datetime column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        datetime_column (str): Column name to sort by.

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    logger.info("Sorting DataFrame by column '%s'.", datetime_column)
    return df.sort_values(datetime_column).reset_index(drop=True)


def create_lag_feature(df: pd.DataFrame, target_column: str, lags: int) -> pd.DataFrame:
    """
    Creates lag features for the specified target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name for which to create lag features.
        lags (int): Number of lag features to create.

    Returns:
        pd.DataFrame: DataFrame with lag features.
    """
    logger.info("Creating %d lag features for column '%s'.", lags, target_column)
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df[target_column].shift(i)
    return df


def create_rolling_mean_feature(
    df: pd.DataFrame, target_column: str, window: int
) -> pd.DataFrame:
    """
    Creates rolling mean features for the specified target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name for which to create rolling mean features.
        window (int): Window size for the rolling mean.

    Returns:
        pd.DataFrame: DataFrame with rolling mean features.
    """
    logger.info(
        "Creating rolling mean feature with window size %d for column '%s'.",
        window,
        target_column,
    )
    df[f"rolling_mean_{window}"] = df[target_column].rolling(window=window).mean()
    return df


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with NaN values and resets the index.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with NaN values removed.
    """
    logger.info("Dropping rows with NaN values.")
    return df.dropna().reset_index(drop=True)


def convert_column_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Converts a specified column to datetime format.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to convert.

    Returns:
        pd.DataFrame: DataFrame with the converted column.
    """
    logger.info("Converting column '%s' to datetime.", column)
    df[column] = pd.to_datetime(df[column])
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using LabelEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    logger.info("Encoding categorical columns using LabelEncoder.")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        logger.debug("Encoding column '%s'.", col)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    return df


def split_data(
    df: pd.DataFrame, train_end_date: str, test_end_date: str, datetime_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and testing datasets based on date.

    Args:
        df (pd.DataFrame): Input DataFrame.
        train_end_date (str): End date for the training set.
        test_end_date (str): End date for the testing set.
        datetime_column (str): Column name with datetime values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    logger.info("Splitting data into train and test sets.")
    train = df[df[datetime_column] <= train_end_date]
    test = df[
        (df[datetime_column] > train_end_date) & (df[datetime_column] <= test_end_date)
    ]
    logger.info("Train set size: %d, Test set size: %d.", len(train), len(test))
    return train, test


def apply_winsorize(
    df: pd.DataFrame, column: str, new_column_name: str, lower: float, upper: float
) -> pd.DataFrame:
    """
    Applies winsorization to a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to winsorize.
        new_column_name (str): Column name for the winsorized column.
        lower (float): Lower percentile for clipping.
        upper (float): Upper percentile for clipping.

    Returns:
        pd.DataFrame: DataFrame with winsorized column.
    """
    logger.info(
        "Applying winsorization to column '%s' with lower: %.2f and upper: %.2f percentiles.",
        column,
        lower,
        upper,
    )
    lower_limit = df[column].quantile(lower)
    upper_limit = df[column].quantile(upper)
    df[new_column_name] = df[column].clip(lower=lower_limit, upper=upper_limit)
    logger.debug(
        "Winsorized column '%s': lower_limit=%.2f, upper_limit=%.2f",
        column,
        lower_limit,
        upper_limit,
    )
    return df


def prepare_training_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str,
    datetime_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepares training and testing data by selecting features and splitting into X and y.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        target_column (str): Name of the target column.
        datetime_column (str): Name of the datetime column to exclude from features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: X_train, y_train, X_test, y_test
    """
    # Identify feature columns
    features = [
        col for col in train_data.columns if col not in [target_column, datetime_column]
    ]

    # Split data into features (X) and target (y)
    X_train, y_train = train_data[features], train_data[target_column]
    X_test, y_test = test_data[features], test_data[target_column]

    return X_train, y_train, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: dict,
    random_state: int,
    target_column: str,
    prediction_column: str,
) -> Tuple[LGBMRegressor, pd.DataFrame]:
    """
    Trains a LightGBM model and generates predictions on the test data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        best_params (dict): Hyperparameters for LightGBM.
        random_state (int): Random state for reproducibility.
        target_column (str): Column name for the target variable.
        prediction_column (str): Column name to store predictions in the test DataFrame.

    Returns:
        Tuple[LGBMRegressor, pd.DataFrame]: Trained model and test data with predictions.
    """
    logger.info("Starting model training with parameters: %s", best_params)

    logger.info("Training LGBMRegressor model.")
    model = LGBMRegressor(**best_params, verbose=-1, random_state=random_state)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
    )

    logger.info("Predicting on test data.")
    predictions = model.predict(X_test)

    comparison_df = pd.DataFrame(
        {prediction_column: predictions, target_column: y_test}
    ).reset_index(drop=True)

    logger.info("Model training complete.")
    return model, comparison_df


def evaluate_model(
    comparison_df: pd.DataFrame, target_column: str, prediction_column: str
) -> Tuple[float, float]:
    """
    Evaluates the model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

    Args:
        comparison_df (pd.DataFrame): DataFrame with actual and predicted values.
        target_column (str): Column with actual target values.
        prediction_column (str): Column with predicted values.

    Returns:
        Tuple[float, float]: RMSE and MAE scores.
    """
    logger.info("Evaluating model performance.")
    rmse = root_mean_squared_error(
        comparison_df[target_column], comparison_df[prediction_column]
    )
    mae = mean_absolute_error(
        comparison_df[target_column], comparison_df[prediction_column]
    )
    logger.info("Evaluation results - RMSE: %.4f, MAE: %.4f.", rmse, mae)
    return rmse, mae
