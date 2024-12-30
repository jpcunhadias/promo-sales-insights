import pytest
import pandas as pd
import numpy as np
from src.promo_sales_insights.pipelines.data_processing.nodes import (
    extract_date_from_code,
    convert_columns_to_string,
    adjust_separator,
    round_columns,
    fill_na_with_median,
    calculate_desconto_percentual,
    calculate_faixa_receita,
    calculate_performance,
    calculate_desconto_percentual_medio,
    calculate_vlr_venda_baseline,
)


def test_extract_date_from_code():
    data = pd.DataFrame({"code": [202301, 202302]})
    new_columns = {
        "cod_ano": "year",
        "semana": "week",
        "data_inicio_semana": "start_date",
    }
    result = extract_date_from_code(data, "code", new_columns)
    assert "year" in result.columns
    assert "week" in result.columns
    assert "start_date" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["start_date"])


def test_convert_columns_to_string():
    data = pd.DataFrame({"col1": [1, 2, 3]})
    result = convert_columns_to_string(data, ["col1"])
    assert result["col1"].dtype == "object"


def test_adjust_separator():
    data = pd.DataFrame({"col": ["1.2", "3.4"]})
    result = adjust_separator(data, "col", ".", "/")
    assert all(result["col"] == ["1/2", "3/4"])


def test_round_columns():
    data = pd.DataFrame({"col": [1.234, 2.345]})
    result = round_columns(data, ["col"], 1)
    assert all(result["col"] == [1.2, 2.3])


def test_fill_na_with_median():
    data = pd.DataFrame({"col": [1, np.nan, 3]})
    result = fill_na_with_median(data, ["col"])
    assert result["col"].isna().sum() == 0


def test_calculate_desconto_percentual():
    data = pd.DataFrame({"price": [100, 200], "discounted_price": [90, 180]})
    result = calculate_desconto_percentual(
        data, ["discounted_price", "price"], "desconto_percentual"
    )
    assert "desconto_percentual" in result.columns
    expected = [0.1, 0.1]
    assert np.allclose(result["desconto_percentual"], expected, atol=1e-6)


def test_calculate_faixa_receita():
    data = pd.DataFrame({"revenue": [10, 20, 30]})
    bins = [0, 15, 25]
    labels = ["low", "medium", "high"]
    result = calculate_faixa_receita(data, "revenue", bins, labels, "faixa_receita")
    assert "faixa_receita" in result.columns
    assert all(result["faixa_receita"].isin(labels))


def test_calculate_desconto_percentual_medio():
    data = pd.DataFrame({"group": ["A", "A", "B"], "discount": [0.1, 0.2, 0.3]})
    result = calculate_desconto_percentual_medio(data, ["group"], "discount")
    assert "discount_medio" in result.columns
    assert np.isclose(result["discount_medio"].iloc[0], 0.15)


def test_calculate_vlr_venda_baseline():
    data = pd.DataFrame({"price": [100, 200], "discount": [0.1, 0.2]})
    result = calculate_vlr_venda_baseline(
        data, ["price", "discount"], "vlr_venda_baseline"
    )
    assert "vlr_venda_baseline" in result.columns
    assert all(result["vlr_venda_baseline"] == [90, 160])
