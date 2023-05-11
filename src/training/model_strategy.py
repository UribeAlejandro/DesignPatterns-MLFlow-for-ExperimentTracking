#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
# Created By: Uribe
# Created Date: 3/5/23
# --------------------------------------------------------------------------- #
# ** Description **
""""""

# Standard Library Imports
import pickle

# --------------------------------------------------------------------------- #
# ** Required libraries **
from abc import ABC, abstractmethod
from typing import Tuple, Union

# Third Party Imports
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential

__all__ = ["LogisticRegressionModel", "RandomForestModel", "NeuralNetwork", "LightGBM"]


class Model(ABC):
    """The Model interface defines the methods that must be implemented by any
    concrete class.

    The module is designed to be flexible and adaptable to different use
    cases and machine learning models. It utilizes the `strategy design
    pattern` to allow users to switch between different machine learning
    models and algorithms in real-time, without modifying the underlying
    code. This is achieved by implementing the `Model Interface`, which
    defines a set of methods and attributes that all concrete
    implementations of the module must adhere.
    """

    @abstractmethod
    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the input data."""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fits the model on the given data."""
        ...

    @property
    @abstractmethod
    def model(self) -> Union[LogisticRegression, RandomForestClassifier]:
        """Returns the underlying model."""
        ...


class LogisticRegressionModel(Model):
    """Implementation of the Model interface using a Logistic Regression.

    Parameters
    ----------
    *args : list
        Variable length argument list.

    **kwargs : dict
        Arbitrary keyword arguments.


    Attributes
    ----------
    model : LogisticRegression
        The underlying LogisticRegression model used for prediction.

    scaler : StandardScaler
        The underlying Standard Scaler.

    Methods
    -------
    preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]
        Preprocesses the data.

    fit(X: pandas.DaraFrame, y: pandas.Series) -> None
        Fits the model on the given data.

    predict(X: pyarrow.Table) -> np.array
        Predicts `adversarial labels` for the given data.

    static scale_data(data_train: pyarrow.Table, data_test: pyarrow.Table, features:
    List[str]) -> Tuple[pyarrow.Table, pyarrow.Table]
        Scales the data by subtracting the mean and dividing it by the standard
        deviation.
    """

    def __init__(self, *args, **kwargs):
        self.__model = LogisticRegression(*args, **kwargs)
        self.__scaler = StandardScaler()
        self.__path_scaler = "scaler/scaler.pkl"

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the input data by scaling and selecting the specified
        features. This method calls the static method `scale_data` to scale the
        data.

        Parameters
        -----------
        X : pd.DataFrame
            The training data as a PyArrow Table.

        y : pd.Series
            The list of features to use in the model.

        Returns
        --------
        Tuple[pd.DataFrame, pd.Series]
          A tuple of the preprocessed data.
        """
        X = self.__scale_data(X)

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fits the model on the given data.

        Parameters
        -----------
        X : pd.DataFrame
            The input features as a Pandas DataFrame.

        y : pd.Series
            The target variable as a Pandas Series.

        Returns
        --------
        None
          None
        """
        mlflow.sklearn.autolog()

        with mlflow.start_run() as run:  # noqa
            self.__model.fit(X, y)
            mlflow.log_artifact(self.__path_scaler)

    def __scale_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Z-scales data.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be scaled.

        Returns
        -------
        pd.DataFrame
            Scaled data.
        """
        X = self.__scaler.fit_transform(X)

        with open(self.__path_scaler, "wb") as file:
            pickle.dump(self.__scaler, file)

        return X

    @property
    def model(self) -> LogisticRegression:
        """Returns the underlying model.

        Returns
        --------
        LogisticRegression
          Underlying model.
        """

        return self.__model

    @property
    def scaler(self) -> StandardScaler:
        """Returns the underlying scaler.

        Returns
        -------
        StandardScaler
            Underlying scaler.
        """
        return self.__scaler


class RandomForestModel(Model):
    """A concrete implementation of the Model interface that uses a Random
    Forest classifier.

    Parameters
    ----------
    *args : list
        Variable length argument list.

    **kwargs : dict
        Arbitrary keyword arguments.

    Attributes
    ----------
    model : LogisticRegression
        The underlying RandomForestClassifier model used for prediction.

    Methods
    --------
    preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]
        Preprocesses the data.

    fit(self, X: pd.DataFrame, y: pd.Series) -> None
        Fits the model on the given data.

    predict(self, X: pd.DataFrame) -> np.array
        Predicts the adversarial labels for the given input features.
    """

    def __init__(self, *args, **kwargs):
        self.__model = RandomForestClassifier(*args, **kwargs)

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the training and tests data by scaling and encoding the
        specified features.

        Parameters
        -----------
        X : pd.DataFrame
            The training data to preprocess.

        y : pd.Series
            The list of feature column names to preprocess.

        Returns
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the preprocessed training and tests data.
        """

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Fits the adversarial model on the training data.

        Parameters
        -----------
        X : pd.DataFrame
            The training features to fit the adversarial model on.

        y : pd.Series
            The corresponding labels to fit the adversarial model on.

        Returns
        --------
        None
            None
        """
        mlflow.sklearn.autolog()

        with mlflow.start_run() as run:  # noqa
            self.__model.fit(X, y, **kwargs)

    @property
    def model(self) -> RandomForestClassifier:
        """Returns the underlying model.

        Returns
        --------
        RandomForestClassifier
          Underlying model.
        """
        return self.__model


class NeuralNetwork(Model):
    """A concrete implementation of the Model interface that uses a Neural
    Network for classification.

    Parameters
    ----------
    *args : list
        Variable length argument list.

    **kwargs : dict
        Arbitrary keyword arguments.

    Attributes
    ----------
    model : LogisticRegression
        The underlying RandomForestClassifier model used for prediction.

    Methods
    --------
    preprocess(self, data_train: pa.Table, data_test: pa.Table, features: List[str])
    -> Tuple[pa.Table, pa.Table]
        Preprocesses the training and tests data by selecting the features and converts
        it to Pandas DataFrames.

    fit(self, X: pd.DataFrame, y: pd.Series) -> None
        Fits the Random Forest model to the training data.

    predict(self, X: pd.DataFrame) -> np.array
        Predicts the adversarial labels for the given input features.
    """

    def __int__(self, *args, **kwargs):
        self.__model = self.__create_model(*args, **kwargs)

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the input data by scaling and selecting the specified
        features. This method calls the static method `scale_data` to scale the
        data.

        Parameters
        -----------
        X : pd.DataFrame
            The training data as a PyArrow Table.

        y : pd.Series
            The list of features to use in the model.

        Returns
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
          A tuple of the preprocessed training and test data.
        """

        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        mlflow.tensorflow.autolog()
        with mlflow.start_run() as run:  # noqa
            self.__model.fit(X, y, **kwargs)

    @staticmethod
    def __create_model(*args, **kwargs):
        input_shape = kwargs.get("input_shape")
        optimizer = kwargs.get("optimizer", "adam")
        loss = kwargs.get("loss", "binary_crossentropy")
        metrics = kwargs.get("metrics", ["accuracy", "f1_score"])

        input_layer = [Input(shape=input_shape)]
        hidden_layers = [
            Dense(dim_layer, kernel_initializer=kernel_init, activation=activation)
            for dim_layer, kernel_init, activation in args
        ]
        output_layer = [Dense(1, activation="sigmoid")]

        layers = input_layer + hidden_layers + output_layer

        model = Sequential(layers)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    @property
    def model(self) -> Sequential:
        """Returns the underlying model.

        Returns
        -------
        Sequential
            Underlying model.
        """
        return self.__model


class LightGBM(Model):
    """A concrete implementation of the Model interface that uses a LightGBM
    classifier.

    Parameters
    ----------
    *args : list
        Variable length argument list.

    **kwargs : dict
        Arbitrary keyword arguments.

    Attributes
    ----------
    model : LogisticRegression
        The underlying RandomForestClassifier model used for prediction.

    Methods
    --------
    preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]
        Preprocesses the data.

    fit(self, X: pd.DataFrame, y: pd.Series) -> None
        Fits the Random Forest model to the training data.
    """

    def __int__(self, *args, **kwargs):
        self.__model = LGBMClassifier()

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the input data by scaling and selecting the specified
        features. This method calls the static method `scale_data` to scale the
        data.

        Parameters
        -----------
        X : pd.DataFrame
            The training data as a PyArrow Table.

        y : pd.Series
            The list of features to use in the model.

        Returns
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
          A tuple of the preprocessed training and test data.
        """
        return X, y

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        mlflow.tensorflow.autolog()
        with mlflow.start_run() as run:  # noqa
            self.__model.fit(X, y)

    @property
    def model(self) -> LGBMClassifier:
        """Returns the underlying model.

        Returns
        -------
        LGBMClassifier
            Underlying model.
        """
        return self.__model
