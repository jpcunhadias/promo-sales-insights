"""
This is a boilerplate test file for pipeline 'data_processing'
generated using Kedro 0.19.10.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
from src.promo_sales_insights.pipelines.data_processing.nodes import (
    extract_date_from_code, convert_columns_to_string, fill_na_with_median,
    calculate_desconto_percentual, calculate_faixa_receita, calculate_performance,
    calculate_desconto_percentual_medio, calculate_vlr_venda_baseline
)

def test_extract_date_from_code_extracts_correct_dates():
    df = pd.DataFrame({'code': [202101, 202152, 202201]})
    new_columns = {'cod_ano': 'year', 'semana': 'week', 'data_inicio_semana': 'start_date'}
    result = extract_date_from_code(df, 'code', new_columns)
    assert result['year'].iloc[0] == 2021
    assert result['week'].iloc[1] == 52
    assert result['start_date'].iloc[2] == pd.Timestamp('2022-01-01')

def test_convert_columns_to_string_converts_columns():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = convert_columns_to_string(df, ['A', 'B'])
    assert result['A'].dtype == object
    assert result['B'].dtype == object

def test_fill_na_with_median_fills_na_values():
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
    result = fill_na_with_median(df, ['A', 'B'])
    assert result['A'].iloc[1] == 2
    assert result['B'].iloc[2] == 4.5

def test_calculate_desconto_percentual():
    df = pd.DataFrame({'price': [100, 200], 'discounted_price': [90, 180]})
    result = calculate_desconto_percentual(df, ['discounted_price', 'price'], 'discount_percent')
    assert np.isclose(result['discount_percent'].iloc[0], 0.1)
    assert np.isclose(result['discount_percent'].iloc[1], 0.1)

def test_calculate_desconto_percentual_medio():
    df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'discount': [0.1, 0.2, 0.3, 0.4]
    })
    result = calculate_desconto_percentual_medio(df, ['category'], 'discount')
    assert np.isclose(result['discount_medio'].iloc[0], 0.15)
    assert np.isclose(result['discount_medio'].iloc[2], 0.35)

def test_calculate_vlr_venda_baseline():
    df = pd.DataFrame({
        'price': [100, 200],
        'discount_percent': [0.1, 0.2]
    })
    result = calculate_vlr_venda_baseline(df, ['price', 'discount_percent'], 'baseline_value')
    assert result['baseline_value'].iloc[0] == 90
    assert result['baseline_value'].iloc[1] == 160