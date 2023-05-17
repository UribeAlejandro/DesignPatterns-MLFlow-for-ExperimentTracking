import os.path
from pathlib import Path
from typing import Tuple, Union

from numpy import nan, uint8
from pandas import DataFrame, Series, concat, read_parquet
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer

__all__ = ["pipeline", "extract_data", "transform_data", "load_data"]


def pipeline(
    dataset_id: Union[str, Path],
    data_home: Union[str, Path],
    output_path: Union[str, Path],
) -> Tuple[DataFrame, Series]:
    """Performs a data pipeline operation on the given dataset.

    Parameters
    ----------
    dataset_id : Union[str, Path]
        The identifier of the dataset.

    data_home: Union[str, Path]
        The directory path where the dataset will be located.

    output_path : Union[str, Path]
        The path where the processed data will be saved.

    Returns
    -------
    Tuple[DataFrame, Series]
        A tuple containing the features DataFrame and the target Series.
    """
    if not os.path.isfile(output_path):
        features, target = extract_data(dataset_id, data_home)
        features, target = transform_data(features, target)
        data = concat([features, target], axis=1)
        load_data(data, output_path)
    else:
        raw_data = read_parquet(output_path, engine="pyarrow")
        features = raw_data.drop(["target"], axis=1)
        target = raw_data["target"]

    return features, target


def extract_data(dataset_id: Union[str, Path], data_home: Union[str, Path]) -> Tuple[DataFrame, Series]:
    """Extracts dataset from openml.

    Parameters
    ----------
    dataset_id : Union[str, Path]
        The identifier of the dataset.

    data_home: Union[str, Path]
        The directory path where the dataset will be located.

    Returns
    -------
    Tuple[DataFrame, Series]
        Raw features and target.
    """
    features, target = fetch_openml(
        dataset_id,
        cache=False,
        data_home=data_home,
        return_X_y=True,
        as_frame=True,
        parser="auto",
        version=1,
    )
    return features, target


def transform_data(features: DataFrame, target: Series) -> Tuple[DataFrame, Series]:
    """Apply transformations to raw data.

    Parameters
    ----------
    features : DataFrame
        Raw features.

    target : Series
        Raw labels.

    Returns
    -------
    Tuple[DataFrame, Series]
        Transformed data.
    """
    impute = SimpleImputer(missing_values=nan, strategy="mean")
    features = DataFrame(impute.fit_transform(features), columns=features.columns)

    target = target.astype(uint8)
    target = target.rename("target")

    return features, target


def load_data(data: DataFrame, output_path: Union[str, Path]) -> None:
    """Loads data to specified path.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the data to be loaded.

    output_path : Union[str, Path]
        The path where the data will be saved. It can be a string or a Path object.

    Returns
    -------
    None
    """
    data.to_parquet(output_path, index=False, engine="pyarrow")
