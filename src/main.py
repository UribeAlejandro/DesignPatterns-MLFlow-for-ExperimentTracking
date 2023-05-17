from pathlib import Path

import pandas as pd
import yaml
from yaml import SafeLoader

from src.data.etl import pipeline
from src.training.loop import Pipeline
from src.training.model_strategy import LogisticRegressionModel
from src.utils.command_line import create_parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    yaml_filepath = Path(args.filepath)
    dataset_id = args.dataset
    data_home = args.location
    output_path = Path(data_home, dataset_id).with_suffix(".parquet")

    features, labels = pipeline(dataset_id, data_home, output_path)

    with open(yaml_filepath, "r") as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    kwargs = config_dict["LogisticRegressionModel"]

    # kwargs = config_dict.get("kwargs")
    #
    # for model_name, values in config_dict:
    #     fine_tune = values.get("fine_tune", False)
    #     kwargs = values.get("kwargs")
    #
    #     model = eval(model_name(**kwargs))

    model = LogisticRegressionModel()
    training_pipe = Pipeline(model)

    X_y = pd.concat([features, labels], axis=1).sample(n=100)
    features = X_y.drop(["target"], axis=1)
    labels = X_y["target"]

    training_pipe.training_loop(features, labels, **kwargs)
