export MLFLOW_PORT=5000 &&
export MLFLOW_ARTIFACT_ROOT=mlruns
export MLFLOW_TRACKING_URI=sqlite:///database/mlruns.db &&
mlflow server --backend-store-uri=$MLFLOW_TRACKING_URI --default-artifact-root=file:$MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port $MLFLOW_PORT
