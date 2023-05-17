from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

from src.constants import CV, RANDOM_STATE


def fine_tuner_bayes(estimator: Any, X: pd.DataFrame, y: pd.Series, param_grid: dict) -> BayesSearchCV:
    """Fine-tune a machine learning model using Bayesian optimization.

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
    BayesSearchCV
        The trained machine learning model.
    """
    cv = BayesSearchCV(estimator=estimator, search_spaces=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE)
    return cv.fit(X, y)


def fine_tuner_randomized(estimator: Any, X: pd.DataFrame, y: pd.Series, param_grid: dict) -> RandomizedSearchCV:
    """Fine-tunes a model using randomized search.

    Parameters
    ----------
    estimator : Model
        The model to fine-tune.

    X : pd.DataFrame
        The training data.

    y : pd.Series
        The target values for the training data.

    param_grid : dict
        The hyperparameters to search over.

    Returns
    -------
    RandomizedSearchCV
        The fine-tuned model.
    """
    cv = RandomizedSearchCV(
        estimator=estimator, param_distributions=param_grid, cv=CV, n_jobs=-1, random_state=RANDOM_STATE
    )
    return cv.fit(X, y)
