"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    extract_date_from_code,
    convert_columns_to_string,
    fill_na_with_median,
    calculate_desconto_percentual,
    calculate_performance,
    calculate_faixa_receita,
    calculate_desconto_percentual_medio,
    calculate_vlr_venda_baseline,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                extract_date_from_code,
                inputs=[
                    "sales",
                    "params:columns_to_create_dates",
                    "params:derived_time_columns",
                ],
                outputs="sales_with_dates",
                name="extract_date_from_code",
            ),
            node(
                convert_columns_to_string,
                inputs=["sales_with_dates", "params:columns_to_convert_to_string"],
                outputs="sales_with_dates_string",
                name="convert_columns_to_string",
            ),
            node(
                fill_na_with_median,
                inputs=["sales_with_dates_string", "params:columns_to_fill_na"],
                outputs="treated_sales_data_initial",
                name="fill_na_with_median",
            ),
            node(
                calculate_desconto_percentual,
                inputs=[
                    "treated_sales_data_initial",
                    "params:features_to_create.desconto_percentual.columns_to_use",
                    "params:features_to_create.desconto_percentual.name",
                ],
                outputs="treated_sales_data_with_desconto_percentual",
                name="calculate_desconto_percentual",
            ),
            node(
                calculate_faixa_receita,
                inputs=[
                    "treated_sales_data_with_desconto_percentual",
                    "params:features_to_create.faixa_receita.column_to_use",
                    "params:features_to_create.faixa_receita.bins",
                    "params:features_to_create.faixa_receita.labels",
                    "params:features_to_create.faixa_receita.name",
                ],
                outputs="treated_sales_data_with_faixa_receita",
                name="calculate_faixa",
            ),
            node(
                calculate_performance,
                inputs=[
                    "treated_sales_data_with_faixa_receita",
                    "params:features_to_create.performance.columns_to_use",
                    "params:features_to_create.performance.labels",
                    "params:features_to_create.performance.name",
                    "params:features_to_create.performance.columns_to_merge_on",
                    "params:features_to_create.performance.how",
                ],
                outputs="treated_sales_data_with_performance",
                name="calculate_performance",
            ),
            node(
                calculate_desconto_percentual_medio,
                inputs=[
                    "treated_sales_data_with_performance",
                    "params:features_to_create.desconto_percentual_medio.columns_to_group_by_and_merge",
                    "params:features_to_create.desconto_percentual_medio.coluna_desconto",
                ],
                outputs="treated_sales_data_with_desconto_percentual_medio",
                name="calculate_desconto_percentual_medio",
            ),
            node(
                calculate_vlr_venda_baseline,
                inputs=[
                    "treated_sales_data_with_desconto_percentual_medio",
                    "params:features_to_create.vlr_venda_baseline.columns_to_use",
                    "params:features_to_create.vlr_venda_baseline.name",
                ],
                outputs="treated_sales_data",
                name="calculate_vlr_venda_baseline",
            )
        ]
    )