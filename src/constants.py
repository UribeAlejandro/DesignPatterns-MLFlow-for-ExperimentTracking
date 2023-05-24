import os

# General
RANDOM_STATE = 42

# Hyperparameter Tune
CV = 5

# Experiment Tracking
MLFLOW_PORT = os.getenv("MLFLOW_PORT")
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT")

# Training
TEST_SIZE = 0.3
MODEL_PATH = "model"
SCALER_FOLDER = "scaler"
SCALER_PATH = f"{MLFLOW_ARTIFACT_ROOT}/scaler.pkl"
# Tensorflow
BATCH_SIZE = 64
BUFFER_SIZE = 1000
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
