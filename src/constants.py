import os

CV = 5
RANDOM_STATE = 42
MLFLOW_PORT = os.getenv("MLFLOW_PORT")
MLFLOW_TRACKING_URI = f"http://127.0.0.1:{MLFLOW_PORT}"
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT")

MODEL_PATH = "model"
SCALER_FOLDER = "scaler"
SCALER_PATH = f"{MLFLOW_ARTIFACT_ROOT}/scaler.pkl"

EXPLAINABILITY_ALGORITHM = "permutation"
# 'exact', 'permutation', 'partition', 'kernel'

TEST_SIZE = 0.3
