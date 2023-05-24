import warnings
from typing import Tuple

import mlflow
import pandas as pd
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.model_selection import train_test_split

from src.constants import MLFLOW_ARTIFACT_ROOT, RANDOM_STATE, TEST_SIZE
from src.training.factory.Model import Model
from src.training.strategy.search_algorithm import SearchAlgorithm
from src.utils.cli import set_logger

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

__all__ = ["Pipeline"]
logger = set_logger(__name__)


class Pipeline:
    """A class for training and evaluating machine learning models.

    Parameters
    ----------
    model : Model
        The machine learning algorithm to be used.

    Attributes
    ----------
    model : Model
        The trained machine learning model.

    search_algorithm : SearchAlgorithm
        Search Algorithm
    """

    def __init__(self, model: Model, search_algorithm: SearchAlgorithm):
        self.__model = model
        self.__search_algorithm = search_algorithm

    def train(self, X: pd.DataFrame, y: pd.Series, experiment_name: str, **kwargs) -> None:
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
        logger.info("Started: Preprocess data")

        X_train, X_test, y_train, y_test = self.__preprocess(X, y)

        logger.info("Finished: Preprocess data")

        logger.info("Get/Create %s experiment", experiment_name)

        self.__search_create_experiment(experiment_name)

        logger.info("Started: Training")

        self.__train(X_train, X_test, y_train, y_test, **kwargs)

        logger.info("Finished: Training")

    def __train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
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
        self.__model.fit(X_train, X_test, y_train, y_test, self.__search_algorithm, **kwargs)

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=RANDOM_STATE, stratify=y, shuffle=True, test_size=TEST_SIZE
        )

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train, X_test, y_train, y_test = self.__model.preprocess(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    @property
    def model(self) -> Model:
        """Returns the model.

        Returns
        -------
        Model
            The model.
        """
        return self.__model

    @property
    def search_algorithm(self) -> SearchAlgorithm:
        """Returns the search algorithm.

        Returns
        -------
        SearchAlgorithm
            The search algorithm.
        """
        return self.__search_algorithm

    @staticmethod
    def __search_create_experiment(experiment_name: str) -> None:
        """Searches for a MLFlow experiment with the given name and creates it
        if it does not exist.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to search for or create.
        """
        experiment = mlflow.search_experiments(filter_string=f"name='{experiment_name}'")

        if not experiment:
            logger.info("%s: experiment does not exist. It will be created.", experiment_name)
            mlflow.create_experiment(name=experiment_name, artifact_location=MLFLOW_ARTIFACT_ROOT)
        else:
            logger.info("%s: experiment already exists.", experiment_name)

        mlflow.set_experiment(experiment_name)
