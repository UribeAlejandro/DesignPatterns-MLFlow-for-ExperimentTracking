#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
# Created By: Uribe
# Created Date: 10/5/23
# --------------------------------------------------------------------------- #
# ** Description **
""""""

# Standard Library Imports
from typing import Dict, List, Optional, Union

# Third Party Imports
import numpy as np

# --------------------------------------------------------------------------- #
# ** Required libraries **
import pandas as pd
from sklearn.model_selection import train_test_split

# docformatter Package Imports
from src.training.model_strategy import Model


class Pipeline:
    """Monitors the data for concept drift by comparing the performance of the
    model on train and tests data. If drift is detected, the provided
    adversarial model is used to detect and measure the drift.

    Parameters
    ----------
    algorithm : Model
        An object implementing the AdversarialModel interface to detect and measure the
        concept drift.

    data_train : Union[pa.Table, pd.DataFrame]
        The training data used to train the model and monitor for concept drift.

    data_test : Union[pa.Table, pd.DataFrame]
        The tests data used to monitor for concept drift.

    features : Optional[List[str]] (default=None)
        The list of feature names to use in the analysis. If None, all columns except
        the target_column are used.

    target_column : Optional[str] (default=None)
        The name of the target column in the data. If None, is assumed the data comes
        from unsupervised learning.

    shuffle : Optional[bool] (default=False)
        If True, the data is shuffled before training the `Adversarial Model`.

    Methods
    -------
    evaluate_drift(self) -> None
        This method evaluates the concept drift using the adversarial model and prints
        the results.

    explain_drift(self) -> None
        Calculates the feature importance for the trained adversarial model. The
        features with higher importance are likely to be responsible for the drift.

    plot_roc_curve(self) -> None
        This method plots the ROC curve of the adversarial model using the testing data.

    private check_parameters(self) -> None
        This method checks if the parameters passed to the class are valid.

    private generate_adv_data(self) -> Tuple[pd.DataFrame, pd.Series]
        This method generates the adversarial data for the training set and returns it
        as a tuple.

    private train(self, X: pd.DataFrame, y: pd.Series) -> None
        This method trains the adversarial model using the provided training data.
    """

    def __init__(
        self,
        algorithm: Model,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        shuffle: Optional[bool] = False,
    ):
        self.__algorithm = algorithm
        self.__data = data
        self.__features = features
        self.__target_column = target_column
        self.__shuffle = shuffle

        self.__metrics: Dict[str, Union[pd.Series, np.array]]
        self.__X: pd.DataFrame
        self.__y: pd.Series

    def training_loop(self, *args, **kwargs) -> None:
        """Evaluates drift by comparing the AUC of ROC curves of original and
        adversarial tests datasets.

        Returns
        --------
        None
        """

        self.__X = self.__data.drop([self.__target_column], axis=1)
        self.__y = self.__data[self.__target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            self.__X, self.__y, test_size=0.3
        )
        X_train, y_train = self.__algorithm.preprocess(X_train, y_train)
        self.__train(X_train, y_train)

    def __train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the adversarial model on the input training data.

        Parameters
        ----------
        X : pd.DataFrame
          DataFrame containing the training data features.

        y : pd.Series
          Series containing the training data labels.

        Returns
        --------
        None
        """
        self.__algorithm.fit(self.__X, self.__y)
