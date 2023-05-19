from pathlib import Path

import yaml
from yaml import SafeLoader

from src.data.etl import etl_pipeline
from src.training.loop import Pipeline
from src.training.model_strategy import (  # noqa
    LightgbmModel,
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel,
)
from src.utils.miscellaneous import create_parser, set_logger

if __name__ == "__main__":
    logger = set_logger(__name__)
    logger.info("Started")

    parser = create_parser()
    args = parser.parse_args()
    dataset_id = args.dataset
    data_home = args.location
    experiment_name = args.experiment_name

    yaml_filepath = Path(args.filepath)
    output_path = Path(data_home, dataset_id).with_suffix(".parquet")

    with open(yaml_filepath, "r") as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    logger.info("Started: ETL")
    features, labels = etl_pipeline(dataset_id, data_home, output_path)
    logger.info("Finished: ETL")

    logger.info("Started: Training Loop")
    for model_name, kwargs in config_dict.items():
        kwargs_creation = kwargs.get("creation", {})
        try:
            model = eval(f"{model_name}(**kwargs_creation)")
        except Exception as e:
            logger.error("%s is not implemented", model_name)
            raise NotImplementedError(e)

        logger.info("Started: Training Loop - %s", model_name)
        training_pipe = Pipeline(model)
        training_pipe.training_loop(features, labels, experiment_name, **kwargs)
        logger.info("Finished: Training Loop - %s", model_name)

    logger.info("Finished: Training Loop")
