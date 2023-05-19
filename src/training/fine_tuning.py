from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from src.constants import CV, RANDOM_STATE

__all__ = ["BayesSearch", "RandomSearch", "GridSearch"]


class FineTuner(ABC):
    """Fine-tune a machine learning model using Bayesian Search CV, Randomized
    Search or Grid Search."""

    @abstractmethod
    def fit(self, estimator: Any, X: DataFrame, y: Series, param_grid: dict) -> None:
        """Fits strategy search.

        Parameters
        ----------
        estimator : Model
            The machine learning model to be fine-tuned.

        X : pd.DataFrame
            The training data.

        y : pd.Series
            The target values for the training data.

        param_grid : dict
            A dictionary of hyperparameters to search over.

        Returns
        -------
        Union[BayesSearchCV, RandomizedSearchCV]
            The trained machine learning model.
        """
        ...

    @property
    @abstractmethod
    def search_algorithm(self) -> FineTuner:
        """

        Returns
        -------
        FineTuner
            Instance of search.
        """
        ...


class BayesSearch(FineTuner):
    def __int__(self, **kwargs):
        self.__fine_tuner: BayesSearchCV

    def fit(self, estimator: Any, X: DataFrame, y: Series, param_grid: dict) -> None:
        self.__fine_tuner = BayesSearchCV(
            estimator=estimator, search_spaces=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE
        )

    @property
    def search_algorithm(self) -> FineTuner:
        return self.__fine_tuner


class RandomSearch(FineTuner):
    def __int__(self):
        self.__fine_tuner: RandomizedSearchCV

    def fit(self, estimator: Any, X: DataFrame, y: Series, param_grid: dict) -> None:
        self.__fine_tuner = RandomizedSearchCV(
            estimator=estimator, param_distributions=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE
        )

    @property
    def search_algorithm(self) -> FineTuner:
        return self.__fine_tuner


class GridSearch(FineTuner):
    def __int__(self):
        self.__fine_tuner: GridSearchCV

    def fit(self, estimator: Any, X: DataFrame, y: Series, param_grid: dict) -> None:
        self.__fine_tuner = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=CV, n_jobs=-1)

    @property
    def search_algorithm(self) -> FineTuner:
        return self.__fine_tuner
