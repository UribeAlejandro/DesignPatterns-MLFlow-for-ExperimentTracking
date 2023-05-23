from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Normalization
from keras.models import Sequential
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow import Tensor
from tensorflow.data import Dataset

from src.constants import MLFLOW_TRACKING_URI, MODEL_PATH, RANDOM_STATE, SCALER_FOLDER, SCALER_PATH
from src.training.strategy.search_algorithm import SearchAlgorithm
from src.utils.miscellaneous import set_logger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = set_logger(__name__)

__all__ = [
    "Model",
    "LogisticRegressionModel",
    "RandomForestModel",
    "NeuralNetworkModel",
    "LightGBMModel",
]

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class Model(ABC):
    """An interface for machine learning models.

    Methods
    ----------------
    preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
    y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        Abstract method for preprocessing the data.

    fit(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
    y_test: pd.Series, **kwargs) -> None:
        Abstract method for fitting the model on the given data.

    Attributes
    ----------
    model: Model
        The underlying model.
    """

    @abstractmethod
    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess the data.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.

        X_test : pd.DataFrame
            The test data.

        y_train : pd.Series
            The target values for the training data.

        y_test : pd.Series
            The target values for the test data.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            The preprocessed training data, test data, training target values, and test target values.
        """
        ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fine_tuner: SearchAlgorithm,
        **kwargs,
    ) -> None:
        """Fits the model on the given data.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.

        X_test : pd.DataFrame
            The test data.

        y_train : pd.Series
            The target values for the training data.

        y_test : pd.Series
            The target values for the test data.

        fine_tuner : SearchAlgorithm
            The search algorithm.

        **kwargs
            Additional keyword arguments to pass to the fitting algorithm.
        """
        ...

    @property
    @abstractmethod
    def model(self) -> Model:
        """Returns the underlying model.

        Returns
        -------
        Model
            The underlying model.
        """
        ...


class LogisticRegressionModel(Model):
    """A concrete implementation of the Model interface that uses a Logistic
    Regression."""

    def __init__(self, *args, **kwargs):
        self.__scaler = StandardScaler()
        self.__model = LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1, *args, **kwargs)

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train_scaled = pd.DataFrame(self.__scale_data(X_train), columns=X_train.columns)

        X_test_scaled = pd.DataFrame(self.__scaler.transform(X_test), columns=X_test.columns)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fine_tuner: SearchAlgorithm,
        **kwargs,
    ) -> None:
        mlflow.sklearn.autolog(silent=True)
        with mlflow.start_run(nested=True) as run:
            logger.info("Run ID: %s", run.info.run_id)
            self.__model = _sklearn_loop(self.__model, X_train, X_test, y_train, y_test, fine_tuner, **kwargs)
            mlflow.log_artifact(SCALER_PATH, SCALER_FOLDER)

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

        with open(SCALER_PATH, "wb") as file:
            pickle.dump(self.__scaler, file)

        return X

    @property
    def model(self) -> LogisticRegression:
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
    Forest classifier."""

    def __init__(self, *args, **kwargs):
        self.__model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, *args, **kwargs)

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fine_tuner: SearchAlgorithm,
        **kwargs,
    ) -> None:
        mlflow.sklearn.autolog(silent=True)
        with mlflow.start_run(nested=True) as run:
            logger.info("Run ID: %s", run.info.run_id)
            self.__model = _sklearn_loop(self.__model, X_train, X_test, y_train, y_test, fine_tuner, **kwargs)

    @property
    def model(self) -> RandomForestClassifier:
        return self.__model


class LightGBMModel(Model):
    """A concrete implementation of the Model interface that uses a LightGBM
    classifier."""

    def __init__(self, *args, **kwargs):
        self.__model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, *args, **kwargs)

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fine_tuner: SearchAlgorithm,
        **kwargs,
    ) -> None:
        mlflow.lightgbm.autolog(silent=True)
        with mlflow.start_run(nested=True) as run:
            logger.info("Run ID: %s", run.info.run_id)
            self.__model = _sklearn_loop(self.__model, X_train, X_test, y_train, y_test, fine_tuner, **kwargs)

    @property
    def model(self) -> LGBMClassifier:
        return self.__model


class NeuralNetworkModel(Model):
    """A concrete implementation of the Model interface that uses a Neural
    Network for classification."""

    def __init__(self, *args, **kwargs):
        self.__model: Sequential = None
        self.__scaler = StandardScaler()

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train = self.__scale_data(X_train)
        X_test = self.__scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fine_tuner: SearchAlgorithm,
        **kwargs,
    ) -> None:
        kwargs_fit = kwargs.get("fit", {})
        kwargs_creation = kwargs.get("creation", {})
        kwargs_compilation = kwargs.get("compilation", {})

        model = self.__create_model(X_train, **kwargs_creation)
        model = self.__compile_model(model, **kwargs_compilation)
        training_batches, test_batches = self.__create_datasets(X_train, X_test, y_train, y_test)

        mlflow.tensorflow.autolog(silent=True)
        with mlflow.start_run(nested=True) as run:
            logger.info("Run ID: %s", run.info.run_id)
            self.__model = model.fit(training_batches, validation_data=test_batches, **kwargs_fit)
            mlflow.log_artifact(SCALER_PATH, SCALER_FOLDER)

    @staticmethod
    def __create_model(
        X_train: Tensor, layers_units: List[int], activation: str, dropout: Optional[float] = 0.3
    ) -> Sequential:
        """Creates a sequential model.

        Parameters
        ----------
        X_train : Tuple[int, int]
            The input dimensions of the model.

        layers_units : List[int]
            The number of units in each layer of the model.

        activation : str
            The activation function to use for each layer of the model.

        dropout : Optional[float] (default=0.3)
            Dropout rate.

        Returns
        -------
        Sequential
            The created model.
        """
        normalization = Normalization(axis=-1)
        normalization.adapt(X_train)

        model = Sequential()
        model.add(normalization)

        for units in layers_units:
            model.add(Dense(units, activation=activation))

        model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))
        return model

    @staticmethod
    def __compile_model(model: Sequential, *args, **kwargs) -> Sequential:
        """Compiles a model.

        Parameters
        ----------
        model : Sequential
            The model to compile.

        *args
            Additional arguments to pass to the `compile` method.

        **kwargs
            Additional keyword arguments to pass to the `compile` method.

        Returns
        -------
        Sequential
            The compiled model.
        """
        model.compile(*args, **kwargs)
        return model

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

        with open(SCALER_PATH, "wb") as file:
            pickle.dump(self.__scaler, file)

        return X

    @staticmethod
    def __create_datasets(
        X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, batch_size: int = 32
    ) -> Tuple[Dataset, Dataset]:
        """Create tensorflow datasets for training and testing.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training data.

        X_test : pd.DataFrame
            The test data.

        y_train : pd.Series
            The target values for the training data.

        y_test : pd.Series
            The target values for the test data.

        Returns
        -------
        Tuple[Dataset, Dataset]
            Datasets.
        """
        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)

        y_train = np.asarray(y_train).astype("float32").reshape((-1, 1))
        y_test = np.asarray(y_test).astype("float32").reshape((-1, 1))

        training_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        training_batches = training_ds.shuffle(1000).batch(batch_size)
        test_batches = test_ds.shuffle(1000).batch(batch_size)

        return training_batches, test_batches

    @property
    def model(self) -> LogisticRegression:
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


def _sklearn_loop(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    fine_tuner: SearchAlgorithm,
    **kwargs,
) -> Model:
    """Trains a model using the scikit-learn API.

    Parameters
    ----------
    model : Model
        The model to train.

    X_train : pd.DataFrame
        The training data.

    X_test : pd.DataFrame
        The test data.

    y_train : pd.Series
        The target values for the training data.

    y_test : pd.Series
        The target values for the test data.

    **kwargs
        Additional keyword arguments to pass to the `fit` method.

    Returns
    -------
    Model
        The trained model.
    """
    kwargs_fit = kwargs.get("fit", {})
    kwargs_evaluate = kwargs.get("evaluate", {})
    evaluate_flag = kwargs_evaluate.get("flag", False)

    if fine_tuner:
        logger.info("Started: Fine tuning")

        kwargs_fine_tune = kwargs.get("fine_tune", {})
        param_grid = kwargs_fine_tune.get("param_grid", {})
        fine_tuner.fit_search_algorithm(model, X_train, y_train, param_grid)
        model = fine_tuner.search_algorithm.best_estimator_

        logger.info("Finished: Fine tuning")
    else:
        model.fit(X_train, y_train, **kwargs_fit)

    if evaluate_flag:
        logger.info("Started: Evaluation")

        evaluator_config = kwargs_evaluate.get("evaluator_config")
        eval_data = X_test.copy()
        eval_data["target"] = y_test.copy()

        model_info = mlflow.sklearn.log_model(model, MODEL_PATH)

        mlflow.evaluate(
            model_info.model_uri,
            data=eval_data,
            targets="target",
            model_type="classifier",
            evaluators="default",
            evaluator_config=evaluator_config,
        )

        logger.info("Finished: Evaluation")

    return model
