from __future__ import annotations

from typing import Any

from pandas import DataFrame, Series

from src.training.factory.FineTuner import FineTuner

__all__ = ["SearchAlgorithm"]


class SearchAlgorithm:
    """A class for fine-tuning machine learning models.

    Parameters
    ----------
    search_algo : FineTuner
        A FineTunerFactory object that specifies the search algorithm to be used.

    Attributes
    ----------
    search_algorithm : FineTuner
        The search algorithm to be used.

    cv : FineTuner
        The cross-validation FineTunerFactory trained.

    Methods
    -------
    fit_search_algorithm(estimator, X, y, param_grid)
        Fits the search algorithm to the given data.

    search_algorithm
        Gets the search algorithm.
    """

    def __init__(self, search_algo: FineTuner):
        self.__search_algo = search_algo
        self.__cv: FineTuner = None

    def fit_search_algorithm(self, estimator: Any, X: DataFrame, y: Series, param_grid: dict) -> None:
        """Fits the search algorithm to the given data.

        Parameters
        ----------
            estimator: The estimator to be finetuned.
            X: The training data.
            y: The training labels.
            param_grid: The hyperparameter grid to be searched.
        """
        self.__search_algo.fit(estimator, X, y, param_grid)

    @property
    def search_algorithm(self) -> FineTuner:
        """Gets the search algorithm.

        Returns
        -------
            The search algorithm.
        """
        return self.__search_algo.search_algorithm
