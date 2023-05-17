from __future__ import annotations

import pickle
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential

from src.constants import RANDOM_STATE
from src.training.fine_tuning import fine_tuner_bayes, fine_tuner_randomized

warnings.simplefilter("ignore")
warnings.simplefilter("ignore")

mlflow.set_tracking_uri("sqlite:///data/mlflow/database/mlruns.db")

__all__ = [
    "Model",
    "LogisticRegressionModel",
    "RandomForestModel",
    "NeuralNetworkModel",
    "LightgbmModel",
]


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

    private static scale_data(data_train: pyarrow.Table, data_test: pyarrow.Table, features:
    List[str]) -> Tuple[pyarrow.Table, pyarrow.Table]
        Scales the data by subtracting the mean and dividing it by the standard
        deviation.
    """

    def __init__(self, *args, **kwargs):
        self.__scaler = StandardScaler()
        self.__path_scaler = "data/mlflow/scaler/scaler.pkl"
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
        **kwargs,
    ) -> None:
        kwargs_fit = kwargs.get("fit", {})
        kwargs_fine_tune = kwargs.get("fine_tune", {})

        fine_tune_flag = kwargs_fine_tune.get("flag", False)
        param_grid = kwargs_fine_tune.get("param_grid", {})

        eval_data = X_test.copy()
        eval_data["target"] = y_test.copy()

        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:  # noqa
            if fine_tune_flag:
                print("Here")
                model = fine_tuner_randomized(self.__model, X_train, y_train, param_grid)
                self.__model = model.best_estimator_
            else:
                self.__model.fit(X_train, y_train, **kwargs_fit)

            model_info = mlflow.sklearn.log_model(self.__model, "model")
            mlflow.evaluate(
                model_info.model_uri,
                data=eval_data,
                targets="target",
                model_type="classifier",
                evaluators=["default"],
            )
            mlflow.log_artifact(self.__path_scaler, "scaler")

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
        **kwargs,
    ) -> None:
        fine_tune = kwargs.get("fine_tune", False)
        param_grid = kwargs.get("param_grid")
        kwargs_fit = kwargs.get("fit")

        eval_data = X_test
        eval_data["target"] = y_test

        mlflow.sklearn.autolog()
        with mlflow.start_run() as run:  # noqa
            if fine_tune:
                model = fine_tuner_bayes(self.__model, X_train, y_train, param_grid)
                self.__model = model.best_estimator_

            else:
                self.__model.fit(X_train, y_train, **kwargs_fit)

            mlflow.shap.log_explanation(self.__model.predict, X_train)
            model_info = mlflow.sklearn.log_model(self.__model, "model")
            mlflow.evaluate(
                model_info.model_uri,
                eval_data,
                targets="target",
                model_type="classifier",
                evaluators=["default"],
            )

    @property
    def model(self) -> RandomForestClassifier:
        return self.__model


class NeuralNetworkModel(Model):
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
        kwargs_creation = kwargs.get("creation")
        kwargs_compilation = kwargs.get("compilation")

        __model = self.__create_model(**kwargs_creation)
        __model = self.__compile_model(__model, **kwargs_compilation)

        self.__model = __model

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
        **kwargs,
    ) -> None:
        mlflow.tensorflow.autolog()
        with mlflow.start_run() as run:  # noqa
            self.__model.fit(X_train, y_train, validation_data=(X_test, y_test), **kwargs)
            mlflow.shap.log_explanation(self.__model.predict, X_train)

    @staticmethod
    def __create_model(input_dim: Tuple[int, int], layers_units: List[int], activation: str) -> Sequential:
        """Creates a sequential model.

        Parameters
        ----------
        input_dim : Tuple[int, int]
            The input dimensions of the model.

        layers_units : List[int]
            The number of units in each layer of the model.

        activation : str
            The activation function to use for each layer of the model.

        Returns
        -------
        Sequential
            The created model.
        """
        model = Sequential()
        model.add(Input(shape=input_dim))

        for units in layers_units:
            model.add(Dense(units, activation=activation))

        model.add(Dropout(0.2))
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

    @property
    def model(self) -> Sequential:
        return self.__model


class LightgbmModel(Model):
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
        **kwargs,
    ) -> None:
        fine_tune = kwargs.get("fine_tune", False)
        param_grid = kwargs.get("param_grid")

        eval_data = X_test
        eval_data["target"] = y_test

        mlflow.lightgbm.autolog()
        with mlflow.start_run() as run:  # noqa
            if fine_tune:
                model = fine_tuner_randomized(self.__model, X_train, y_train, param_grid)
                self.__model = model.best_estimator_

            else:
                self.__model.fit(X_train, y_train)

            mlflow.shap.log_explanation(self.__model.predict, X_train)
            model_info = mlflow.sklearn.log_model(self.__model, "model")
            mlflow.evaluate(
                model_info.model_uri,
                eval_data,
                targets="target",
                model_type="classifier",
                evaluators=["default"],
            )

    @property
    def model(self) -> LGBMClassifier:
        return self.__model
