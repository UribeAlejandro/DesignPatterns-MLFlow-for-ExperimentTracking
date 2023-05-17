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

    Methods
    -------
    training_loop(X_train: pd.DataFrame, y_train: pd.Series, **kwargs)
        Train a Model's concrete implementation.
    """

    def __init__(self, algorithm: Model):
        self.__algorithm = algorithm

    def training_loop(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """Train a Model's concrete implementation.

        Parameters
        ----------
        X : pd.DataFrame
            The training data.

        y : pd.Series
            The target values for the training data.

        **kwargs
            Additional keyword arguments to pass to the training algorithm.

        Returns
        -------
        None
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, stratify=y, shuffle=True)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        X_train, X_test, y_train, y_test = self.__algorithm.preprocess(X_train, X_test, y_train, y_test)

        self.__train(X_train, X_test, y_train, y_test, **kwargs)

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

        **kwargs
            Additional keyword arguments to pass to the training algorithm.

        Returns
        -------
        None
        """
        self.__algorithm.fit(X_train, X_test, y_train, y_test, **kwargs)

    @property
    def model(self):
        return self.__algorithm
