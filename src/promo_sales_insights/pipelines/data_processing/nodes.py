"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""

import pandas as pd
import numpy as np
from typing import Literal


def extract_date_from_code(
    data: pd.DataFrame, column: str, new_columns: dict[str]
) -> pd.DataFrame:
    """
    Extract date from code
    """
    cod_ano = new_columns.get("cod_ano")
    semana = new_columns.get("semana")
    data_inicio_semana = new_columns.get("data_inicio_semana")

    if cod_ano is None or semana is None or data_inicio_semana is None:
        raise ValueError(
            "new_columns must have the following keys: cod_ano, semana, data_inicio_semana"
        )

    data[cod_ano] = data[column] // 100
    data[semana] = data[column] % 100

    data[semana] = data[semana].clip(lower=1, upper=52)

    data[data_inicio_semana] = pd.to_datetime(
        data[cod_ano].astype(str) + "-01-01", errors="coerce"
    ) + pd.to_timedelta((data[semana] - 1) * 7, unit="D")

    return data


def convert_columns_to_string(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert columns to string
    """
    for col in columns:
        data[col] = data[col].astype(str)

    return data

def adjust_separator(data: pd.DataFrame, column: str, current_separator: str, new_separator:str) -> pd.DataFrame:
    """
    Adjust separator
    """
    data[column] = data[column].str.replace(current_separator, new_separator)
    return data

def round_columns(data: pd.DataFrame, columns: list[str], decimals: int) -> pd.DataFrame:
    """
    Round columns
    """
    for col in columns:
        data[col] = data[col].round(decimals)
    return data

def fill_na_with_median(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Fill NA values with median
    """
    for col in columns:
        data[col] = data[col].fillna(data[col].median())
    return data


def calculate_desconto_percentual(
    data: pd.DataFrame, columns: list[str], new_column_name: str
) -> pd.DataFrame:
    """
    Calculate desconto percentual
    """
    data[new_column_name] = 1 - (data[columns[0]] / data[columns[1]])
    return data


def calculate_faixa_receita(
    data: pd.DataFrame,
    column_to_use: str,
    bins: list[int],
    labels: list[str],
    new_column_name: str,
) -> pd.DataFrame:
    """
    Calculate faixa de receita
    """
    max_value = np.float64(data[column_to_use].max())
    bins_receita = bins + [float(max_value)]
    data[new_column_name] = pd.cut(
        data[column_to_use], bins=bins_receita, labels=labels
    )

    return data


def calculate_performance(
    data: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    new_column_name: str,
    column_to_merge_on: str,
    how: Literal["left", "right", "inner", "outer"],
) -> pd.DataFrame:
    """
    Calculate performance KPI
    """

    temporal_analysis = (
        data.groupby(columns[0])
        .agg(
            receita_total=(columns[1], "sum"),
            desconto_medio=(columns[2], "mean"),
        )
        .reset_index()
    )

    high_treshold = temporal_analysis["receita_total"].quantile(0.75)
    low_treshold = temporal_analysis["receita_total"].quantile(0.25)

    temporal_analysis[new_column_name] = pd.cut(
        temporal_analysis["receita_total"],
        bins=[-float("inf"), low_treshold, high_treshold, float("inf")],
        labels=labels,
    )

    data = data.merge(temporal_analysis[column_to_merge_on], on=columns[0], how=how)

    return data


def calculate_desconto_percentual_medio(
    data: pd.DataFrame, columns_to_group_by_and_merge: list[str], columns_desconto: str
) -> pd.DataFrame:
    """
    Calculate desconto percentual medio
    """
    desconto_medio = (
        data.groupby(columns_to_group_by_and_merge)[columns_desconto]
        .mean()
        .reset_index()
    )
    data = data.merge(
        desconto_medio, on=columns_to_group_by_and_merge, suffixes=("", "_medio")
    )
    return data


def calculate_vlr_venda_baseline(
    data: pd.DataFrame, columns_to_use: list[str], new_column_name: str
) -> pd.DataFrame:
    """
    Calculate vlr_venda_baseline
    """
    data[new_column_name] = data[columns_to_use[0]] * (1 - data[columns_to_use[1]])
    return data
