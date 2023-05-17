from typing import Any, Union

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from src.constants import CV, RANDOM_STATE

__all__ = ["fine_tune"]


def fine_tune(
    estimator: Any, X: pd.DataFrame, y: pd.Series, param_grid: dict, strategy: str = "randomized"
) -> Union[BayesSearchCV, RandomizedSearchCV]:
    """Fine-tune a machine learning model using Bayesian Search CV or
    randomized search.

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

    strategy : str
        Either Bayesian Search CV or Randomized Search

    Returns
    -------
    Union[BayesSearchCV, RandomizedSearchCV]
        The trained machine learning model.
    """
    if strategy == "bayes":
        cv = BayesSearchCV(estimator=estimator, search_spaces=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE)
    elif strategy == "randomized":
        cv = RandomizedSearchCV(
            estimator=estimator, param_distributions=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE
        )
    else:
        raise NotImplementedError

    return cv.fit(X, y)
