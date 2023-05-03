from pandas import concat, DataFrame, Series
from pathlib import Path
from numpy import uint8
from sklearn.datasets import fetch_openml
from typing import Union, Tuple


def extract_data(dataset_id: Union[str, Path], data_home: Union[str, Path]) -> Tuple[DataFrame, Series]:
    data, target = fetch_openml(dataset_id, cache=False, data_home=data_home, return_X_y=True, as_frame=True)
    return data, target


def transform_data(data: DataFrame, target: Series, ) -> DataFrame:
    target = target.astype(uint8)
    data = concat([data, target], axis=1)
    return data


def load_data(data: DataFrame, output_path: Union[str, Path]) -> None:
    data.to_parquet(output_path, index=False, engine="pyarrow")
