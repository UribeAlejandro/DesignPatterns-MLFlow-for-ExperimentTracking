from pathlib import Path

import yaml
from yaml import SafeLoader

from src.data.etl import pipeline
from src.training.loop import Pipeline
from src.training.model_strategy import LightgbmModel, LogisticRegressionModel, NeuralNetworkModel, RandomForestModel
from src.utils.command_line import create_parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    dataset_id = args.dataset
    data_home = args.location
    experiment_name = args.experiment_name
    output_path = Path(data_home, dataset_id).with_suffix(".parquet")

    yaml_filepath = Path(args.filepath)
    with open(yaml_filepath, "r") as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    features, labels = pipeline(dataset_id, data_home, output_path)

    for model_name, kwargs in config_dict.items():
        if model_name == "LogisticRegressionModel":
            model = LogisticRegressionModel()
        elif model_name == "RandomForestModel":
            model = RandomForestModel()
        elif model_name == "RandomForestModel":
            model = LightgbmModel()
        elif model_name == "NeuralNetworkModel":
            model = NeuralNetworkModel()
        else:
            raise NotImplementedError

    training_pipe = Pipeline(model)
    training_pipe.training_loop(features, labels, experiment_name, **kwargs)
