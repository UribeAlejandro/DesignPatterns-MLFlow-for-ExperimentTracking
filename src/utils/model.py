#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------- #
# Created By: Uribe
# Created Date: 3/5/23
# --------------------------------------------------------------------------- #
# ** Description **
""""""

# Standard Library Imports
# --------------------------------------------------------------------------- #
# ** Required libraries **
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


class Model(ABC):
    """The Model interface defines the methods that must be implemented by any
    concrete class that aims to detect and measure concept drift using
    adversarial models.

    The module is designed to be flexible and adaptable to different use
    cases and machine learning models. It utilizes the `strategy design
    pattern` to allow users to switch between different machine learning
    models and algorithms in real-time, without modifying the underlying
    code. This is achieved by implementing the `Adversarial Model
    Interface`, which defines a set of methods and attributes that all
    concrete implementations of the module must adhere.
    """

    @abstractmethod
    def preprocess(
        self, data: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocesses the input data."""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the adversarial model to the preprocessed features and
        `adversarial labels`."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        """Predicts the `adversarial labels` for the given input features."""
        ...

    @property
    @abstractmethod
    def model(self) -> Union[LogisticRegression, RandomForestClassifier]:
        """Returns the feature importances for the trained adversarial
        model."""
        ...


class LogisticRegressionModel(Model):
    """Implementation of the AdversarialModel interface using a logistic
    regression.

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

    Methods
    -------
    preprocess(data_train: pyarrow.Table, data_test: pyarrow.Table, features: List[str])
    -> Tuple[pa.Table, pa.Table]
        Preprocesses the input data.

    fit(X: pandas.DaraFrame, y: pandas.Series) -> None
        Trains the model on the given data.

    predict(X: pyarrow.Table) -> np.array
        Predicts `adversarial labels` for the given data.

    feature_importances(X_columns: List[str]) -> pandas.Series
        Calculates feature importances for the given data.

    static scale_data(data_train: pyarrow.Table, data_test: pyarrow.Table, features:
    List[str]) -> Tuple[pyarrow.Table, pyarrow.Table]
        Scales the data by subtracting the mean and dividing it by the standard
        deviation.
    """

    def __init__(self, *args, **kwargs):
        self.__model = LogisticRegression(*args, **kwargs)
        self.__scaler = StandardScaler()

    def preprocess(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame, features: List[str]
    ) -> Tuple[pa.Table, pa.Table]:
        """Preprocesses the input data by scaling and selecting the specified
        features. This method calls the static method `scale_data` to scale the
        data.

        Parameters
        -----------
        data_train : pd.DataFrame
          The training data as a PyArrow Table.

        data_test : pd.DataFrame
                    The tests data as a PyArrow Table.

        features : List[str]
                    The list of features to use in the model.

        Returns
        --------
        Tuple[pa.Table, pa.Table]
          A tuple of the preprocessed training and tests data.
        """
        X_train = data_train[features]
        data_train[~X_train]

        data_test = data_test.select(features)

        data_train, data_test = self.scale_data(data_train, data_test, features)

        return data_train, data_test

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the adversarial logistic regression model to the training data.

        Parameters
        -----------
        X : pd.DataFrame
            The input features as a Pandas DataFrame.

        y : pd.Series
            The target variable as a Pandas Series.

        Returns
        --------
        None
          None
        """
        self.__model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predicts the adversarial labels for the given input features.

        Parameters
        -----------
        X : pd.DataFrame
            The input features as a Pandas DataFrame.

        Returns
        --------
        np.array
          A numpy array containing the predicted adversarial labels.
        """
        y_pred = self.__model.predict_proba(X)

        return y_pred

    @property
    def model(self) -> LogisticRegression:
        """Returns the feature importances for the trained adversarial model.

        Parameters
        -----------
        X_columns : list
            The list of column names of the input data.

        Returns
        --------
        pd.Series
          A Pandas Series containing the feature importances.
        """

        return self.__model

    @property
    def scaler(self):
        return self.__scaler


class RandomForest(Model):
    """A concrete implementation of the AdversarialModel interface that uses a
    Random Forest classifier to detect and measure concept drift.

    Parameters
    ----------
    *args : list
        Variable length argument list.

    **kwargs : dict
        Arbitrary keyword arguments.

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

    feature_importance(self, X_columns: list) -> pd.Series
        Returns the feature importances for the trained adversarial model.
    """

    def __init__(self, *args, **kwargs):
        self.__model = RandomForestClassifier(*args, **kwargs)

    def preprocess(
        self, data_train: pa.Table, data_test: pa.Table, features: List[str]
    ) -> Tuple[pa.Table, pa.Table]:
        """Preprocesses the training and tests data by scaling and encoding the
        specified features.

        Parameters
        -----------
        data_train : pa.Table
            The training data to preprocess.

        data_test : pa.Table
            The tests data to preprocess.

        features : List[str]
            The list of feature column names to preprocess.

        Returns
        --------
        Tuple[pa.Table, pa.Table]
            A tuple containing the preprocessed training and tests data.
        """
        data_train = data_train.select(features)
        data_test = data_test.select(features)

        return data_train, data_test

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fits the adversarial model on the training data.

        Parameters
        -----------
        X : pd.DataFrame
            The training features to fit the adversarial model on.

        y : pd.Series
            The corresponding labels to fit the adversarial model on.

        Returns
        --------
        None
            None
        """
        self.__model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predicts the adversarial labels for the given input features.

        Parameters
        -----------
        X : pd.DataFrame
            The input features as a Pandas DataFrame.

        Returns
        --------
        np.array
          A numpy array containing the predicted adversarial labels.
        """
        y_pred = self.__model.predict_proba(X)

        return y_pred

    @property
    def model(self, X_columns: list) -> pd.Series:
        """Returns the feature importances for the trained adversarial model.

        Parameters
        -----------
        X_columns : list
            The list of column names of the input data.

        Returns
        --------
        pd.Series
          A Pandas Series containing the feature importances.
        """
        feat_imp = pd.Series(
            np.squeeze(self.__model.feature_importances_), index=X_columns
        )

        return feat_imp


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
        data_train: Union[pa.Table, pd.DataFrame],
        data_test: Union[pa.Table, pd.DataFrame],
        features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        shuffle: Optional[bool] = False,
    ):
        self.__algorithm = algorithm
        self.__data_train = data_train
        self.__data_test = data_test
        self.__features = features
        self.__target = target_column
        self.__shuffle = shuffle

        self.__metrics: Dict[str, Union[pd.Series, np.array]]
        self.__X: pd.DataFrame
        self.__y: pd.Series

        self.__check_parameters()

    def training_loop(self) -> None:
        """Evaluates drift by comparing the AUC of ROC curves of original and
        adversarial tests datasets.

        Returns
        --------
        None
        """

        self.__data_train, self.__data_test = self.__algorithm.preprocess(
            self.__data_train, self.__data_test, self.__features
        )

        self.__X, self.__y = self.__generate_adv_data()
        self.__train(self.__X, self.__y)

        self.__metrics = {
            "y_pred_proba": (y_pred_proba := self.__algorithm.predict(self.__X)),
            "roc_auc": roc_auc_score(self.__y, y_pred_proba[:, 1]),
            "feature_importance": self.__algorithm.feature_importance(self.__X.columns),
        }

    def explain_drift(self, n_features: Optional[int] = None) -> None:
        """Calculates the feature importance for the trained adversarial model
        and prints the most important features. These features are likely to be
        responsible for the detected drift.

        Parameters
        ----------
        n_features : Optional[int] (default=None)
          Specifies the number of features to be taken into account.

        Returns
        -------
        None
        """
        feature_importance = self.__metrics.get("feature_importance")
        if not n_features:
            n_features = len(feature_importance)
        n_features = min(n_features, len(feature_importance))
        feature_importance.nlargest(n_features).plot(kind="barh")
        plt.title(f"Top {n_features} important features")
        plt.show(block=False)

    def plot_roc_curve(self) -> None:
        """Plots the ROC curve for the adversarial data."""
        y_pred = self.__metrics.get("y_pred_proba")[:, 1]
        roc_auc = self.__metrics.get("roc_auc")
        fpr, tpr, threshold = roc_curve(self.__y, y_pred)
        plt.title("Receiver Operating Characteristic")
        plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show(block=False)

    def __check_parameters(self) -> None:
        """Checks the validity of the input parameters for the DriftMonitor
        instance.

        Returns
        --------
        None
        """
        if not self.__target:
            self.__target = "adv_target"

        if isinstance(self.__data_train, pd.DataFrame):
            self.__data_train = pa.Table.from_pandas(self.__data_train)
        if isinstance(self.__data_test, pd.DataFrame):
            self.__data_train = pa.Table.from_pandas(self.__data_test)

        if self.__target in self.__data_test.column_names:
            self.__data_train = self.__data_train.drop([self.__target])
        if self.__target in self.__data_test.column_names:
            self.__data_test = self.__data_test.drop([self.__target])

        if not self.__features:
            self.__features = self.__data_train.column_names
            self.__features.remove(self.__target)

    def __generate_adv_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generates adversarial data by adding adversarial labels.

        Returns
        --------
        Tuple[pd.DataFrame, pd.Series]
          Tuple containing adversarial tests data features and adversarial labels.
        """

        zero_array = pa.array([0] * len(self.__data_train), type=pa.uint8())
        ones_array = pa.array([1] * len(self.__data_test), type=pa.uint8())

        self.__data_train = self.__data_train.add_column(1, self.__target, zero_array)
        self.__data_test = self.__data_test.add_column(1, self.__target, ones_array)

        columns_order = self.__data_train.column_names
        self.__data_test = self.__data_test.select(columns_order)

        adversarial_data = pa.concat_tables([self.__data_train, self.__data_test])
        adversarial_data = adversarial_data.to_pandas()

        if self.__shuffle:
            adversarial_data = adversarial_data.sample(frac=1).reset_index(drop=True)

        X = adversarial_data.drop([self.__target], axis=1)
        y = adversarial_data.loc[:, self.__target]

        return X, y

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
