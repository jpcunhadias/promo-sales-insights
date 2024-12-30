"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    convert_column_to_str,
    sort_by_date,
    create_lag_feature,
    create_rolling_mean_feature,
    dropna,
    convert_column_to_datetime,
    encode_categorical,
    split_data,
    apply_winsorize,
    train_model,
    evaluate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=convert_column_to_str,
            inputs=["treated_sales_data", "params:column_to_convert_to_str"],
            outputs="treated_sales_data_converted",
            name="convert_column_to_str"
        ),
        node(
            func=sort_by_date,
            inputs=["treated_sales_data_converted", "params:datetime_column"],
            outputs="sorted_sales_data",
            name="sort_by_date",
        ),
        node(
            func=create_lag_feature,
            inputs=["sorted_sales_data", "params:target_variable", "params:week_range_for_lags"],
            outputs="sales_data_with_lag",
            name="create_lag_feature",
        ),
        node(
            func=create_rolling_mean_feature,
            inputs=["sales_data_with_lag", "params:target_variable", "params:window_rolling_mean"],
            outputs="sales_data_with_rolling_mean",
            name="create_rolling_mean_feature",
        ),
        node(
            func=dropna,
            inputs="sales_data_with_rolling_mean",
            outputs="sales_data_no_null",
            name="dropna",
        ),
        node(
            func=convert_column_to_datetime,
            inputs=["sales_data_no_null", "params:datetime_column"],
            outputs="sales_data_datetime",
            name="convert_column_to_datetime",
        ),
        node(
            func=encode_categorical,
            inputs="sales_data_datetime",
            outputs="sales_data_encoded",
            name="encode_categorical",
        ),
        node(
            func=split_data,
            inputs=["sales_data_encoded", "params:training_params.train_end_date", "params:training_params.test_end_date", "params:datetime_column"],
            outputs=["train_data", "test_data"],
            name="split_data",
        ),
        node(
            func=apply_winsorize,
            inputs=["train_data", "params:winsorize.column", "params:winsorize.new_column_name", "params:winsorize.lower_bound", "params:winsorize.upper_bound"],
            outputs="train_data_winsorized",
            name="apply_winsorize_train",
        ),
        node(
            func=apply_winsorize,
            inputs=["test_data", "params:winsorize.column","params:winsorize.new_column_name", "params:winsorize.lower_bound", "params:winsorize.upper_bound"],
            outputs="test_data_winsorized",
            name="apply_winsorize_test",
        ),
        node(
            func=train_model,
            inputs=["train_data_winsorized", "test_data_winsorized", "params:training_params.best_params","params:training_params.random_state", "params:target_variable", "params:datetime_column", "params:prediction_column"],
            outputs=["regressor", "test_data_with_predictions"],
            name="train_model",
        ),
        node(
            func=evaluate_model,
            inputs=["test_data_with_predictions", "params:target_variable", "params:prediction_column"],
            outputs=None,
            name="evaluate_model",
        ),
    ])
