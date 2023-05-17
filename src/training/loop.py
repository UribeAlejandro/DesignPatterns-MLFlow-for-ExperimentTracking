from typing import Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import RANDOM_STATE
from src.training.model_strategy import Model

__all__ = ["Pipeline"]


class Pipeline:
    """A class for training and evaluating machine learning models.

    Parameters
    ----------
    algorithm : Model
        The machine learning algorithm to be used.

    Attributes
    ----------
    model : Model
        The trained machine learning model.
    """

    def __init__(self, algorithm: Model):
        self.__algorithm = algorithm

    def training_loop(self, X: pd.DataFrame, y: pd.Series, experiment_name: str, **kwargs) -> None:
        """Train a Model's concrete implementation.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.

        y : pd.Series
            The target values for the training data.

        experiment_name : str
            Underlying MLFlow's experiment name.

        **kwargs
            Additional keyword arguments to pass to the training algorithm.

        Returns
        -------
        None
        """
        self.__search_create_experiment(experiment_name)

        X_train, X_test, y_train, y_test = self.__preprocess(X, y)

        self.__train(X_train, X_test, y_train, y_test, experiment_name, **kwargs)

    def __train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        experiment_name: str,
        **kwargs,
    ) -> None:
        """Train a Model's concrete implementation.

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

        experiment_name : str
            Underlying MLFlow's experiment name.

        **kwargs
            Additional keyword arguments to pass to the training algorithm.

        Returns
        -------
        None
        """
        self.__algorithm.fit(X_train, X_test, y_train, y_test, **kwargs)

    def __preprocess(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocesses the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to preprocess.

        y : pd.Series
            The target values.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, stratify=y, shuffle=True)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train, X_test, y_train, y_test = self.__algorithm.preprocess(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    @property
    def model(self) -> Model:
        """Returns the model.

        Returns
        -------
        Model
            The model.
        """
        return self.__algorithm

    @staticmethod
    def __search_create_experiment(experiment_name: str) -> None:
        """Searches for an experiment with the given name and creates it if it
        does not exist.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to search for or create.
        """
        experiment = mlflow.search_experiments(filter_string=f"name='{experiment_name}'")
        if not experiment:
            mlflow.create_experiment(name=experiment_name, artifact_location="mlruns/")

        mlflow.set_experiment(experiment_name)
