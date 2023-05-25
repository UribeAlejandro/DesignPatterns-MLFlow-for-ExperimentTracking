from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from yaml import SafeLoader

from src.data.etl import etl_pipeline
from src.training.strategy.training import Pipeline
from src.utils.cli import create_parser, set_logger
from src.utils.object_creation import create_fine_tuner, create_model

plt.switch_backend("agg")


def main():
    """Main entry point for the script. It parses the command line arguments,
    loads the configuration file, and trains the models.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger = set_logger(__name__)
    logger.info("Started: Training Loop")

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
        logger.info("Model object creation: %s", model_name)
        model = create_model(model_name, **kwargs)

        logger.info("Fine-tune object creation")
        fine_tuner, kwargs = create_fine_tuner(**kwargs)

        logger.info("Started: Training Loop - %s", model_name)
        training_pipe = Pipeline(model, fine_tuner)
        training_pipe.train(features, labels, experiment_name, **kwargs)
        logger.info("Finished: Training Loop - %s", model_name)

    logger.info("Finished: Training Loop")


if __name__ == "__main__":
    main()
