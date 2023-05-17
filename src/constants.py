from enum import Enum

CV = 5
RANDOM_STATE = 42


class ModelName(Enum):
    LogisticRegressionModel = "LogisticRegressionModel"
    LightgbmModel = "LightgbmModel"
    RandomForestModel = "RandomForestModel"
    NeuralNetworkModel = "NeuralNetworkModel"
