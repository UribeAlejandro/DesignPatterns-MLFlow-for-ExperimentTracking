from typing import Tuple

import numpy as np  # noqa
from skopt.space import Categorical, Integer, Real  # noqa

from src.training.factory.FineTuner import BayesSearch, GridSearch, RandomSearch  # noqa
from src.training.factory.Model import (  # noqa
    LightGBMModel,
    LogisticRegressionModel,
    Model,
    NeuralNetworkModel,
    RandomForestModel,
)
from src.training.strategy.search_algorithm import SearchAlgorithm


def create_fine_tuner(**kwargs) -> Tuple[SearchAlgorithm, dict]:
    """This function creates a fine tuner object from the given keyword
    arguments.

    Parameters
    ----------
    kwargs : dict
        A dictionary of keyword arguments. The following keyword arguments are supported:

    * `fine_tune`: A boolean flag indicating whether to fine tune the model. Defaults to `False`.
    * `param_grid`: A dictionary of hyperparameters to search over. Defaults to an empty dictionary.
    * `strategy`: The search strategy to use. Defaults to `RandomSearch`.

    Returns
    -------
    Tuple[SearchAlgorithm, dict]
        A tuple containing the fine tuner object and the updated keyword arguments.
    """
    kwargs_fine_tune = kwargs.get("fine_tune", {})
    fine_tune_flag = kwargs_fine_tune.get("flag", False)
    param_grid = kwargs_fine_tune.get("param_grid", {})
    strategy = kwargs_fine_tune.get("strategy", "RandomSearch")

    if fine_tune_flag:
        search_algo = eval(f"{strategy}()")
        fine_tuner = SearchAlgorithm(search_algo)
        param_grid = {k: eval(v) for k, v in param_grid.items()}
        kwargs["fine_tune"]["param_grid"] = param_grid
    else:
        fine_tuner = None

    return fine_tuner, kwargs


def create_model(model_name: str, **kwargs) -> Model:
    """This function creates a model object from the given model name and
    keyword arguments.

    Parameters
    ----------
    model_name : str
        The name of the model to create.
    kwargs : dict
        A dictionary of keyword arguments. The following keyword arguments are supported:

    * `creation`: A dictionary of keyword arguments to pass to the model constructor. Defaults to an empty dictionary.

    Returns
    -------
    Model
        The model object.
    """
    kwargs_creation = kwargs.get("creation", {})  # noqa
    try:
        model = eval(f"{model_name}(**kwargs_creation)")
    except Exception as e:
        raise NotImplementedError(f"{e}. {model_name} is not implemented")

    return model
