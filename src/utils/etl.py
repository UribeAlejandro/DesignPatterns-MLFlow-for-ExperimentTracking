#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
# Created By: Uribe
# Created Date: 28/4/23
# --------------------------------------------------------------------------- #
# ** Description **
"""Extract, Transform and Load for datasets from openml."""

# --------------------------------------------------------------------------- #
# ** Required libraries **

# Standard Library Imports
from pathlib import Path
from typing import Tuple, Union

# Third Party Imports
from numpy import uint8
from pandas import DataFrame, Series, concat
from sklearn.datasets import fetch_openml

__all__ = ["extract_data", "transform_data", "load_data"]


def extract_data(
    dataset_id: Union[str, Path], data_home: Union[str, Path]
) -> Tuple[DataFrame, Series]:
    data, target = fetch_openml(
        dataset_id, cache=False, data_home=data_home, return_X_y=True, as_frame=True
    )
    return data, target


def transform_data(
    data: DataFrame,
    target: Series,
) -> DataFrame:
    target = target.astype(uint8)
    data = concat([data, target], axis=1)
    return data


def load_data(data: DataFrame, output_path: Union[str, Path]) -> None:
    data.to_parquet(output_path, index=False, engine="pyarrow")
