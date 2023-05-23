from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from yaml import SafeLoader

from src.data.etl import etl_pipeline
from src.training.factory.fine_tuning import BayesSearch, FineTuner, GridSearch, RandomSearch  # noqa
from src.training.factory.model import (  # noqa
    LightGBMModel,
    LogisticRegressionModel,
    NeuralNetworkModel,
    RandomForestModel,
)
from src.training.loop import Pipeline
from src.utils.miscellaneous import create_parser, set_logger

plt.switch_backend("agg")

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
        kwargs_fine_tune = kwargs.get("fine_tune", {})
        fine_tune_flag = kwargs_fine_tune.get("flag", False)
        try:
            model = eval(f"{model_name}(**kwargs_creation)")
        except Exception as e:
            logger.error("%s is not implemented", model_name)
            raise NotImplementedError(e)

        if fine_tune_flag:
            param_grid = kwargs_fine_tune.get("param_grid", {})
            strategy = kwargs_fine_tune.get("strategy", "RandomSearch")
            search_algo = eval(f"{strategy}()")
            fine_tuner = FineTuner(search_algo)
        else:
            fine_tuner = None

        logger.info("Started: Training Loop - %s", model_name)

        training_pipe = Pipeline(model, fine_tuner)
        training_pipe.train(features, labels, experiment_name, **kwargs)

        logger.info("Finished: Training Loop - %s", model_name)

    logger.info("Finished: Training Loop")
