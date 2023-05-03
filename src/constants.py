from pathlib import Path

DATASET_ID = "higgs"
DATA_LOCATION = "data"
OUTPUT_PATH = Path(DATA_LOCATION, DATASET_ID).with_suffix(".parquet")
