import logging
from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    """This function creates an ArgumentParser object with the following
    arguments:

    -f, --filepath: Path to configuration file.
    -d, --dataset: Dataset name.
    -l, --location: Folder to store data.
    -e, --experiment_name: MLFlow's experiment name.

    Returns
    ------
    ArgumentParser
        The ArgumentParser object.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        action="store",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("-d", "--dataset", action="store", type=str, default="higgs", help="Dataset name")
    parser.add_argument(
        "-l",
        "--location",
        action="store",
        type=str,
        default="data",
        help="Folder to store data",
    )

    parser.add_argument(
        "-e",
        "--experiment_name",
        action="store",
        type=str,
        default="Default",
        help="MLFlow's experiment name",
    )

    return parser


def set_logger(name: str) -> logging.Logger:
    """This function sets up a logger with the given name. The logger will
    write messages to the console.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The logger object.
    """
    logger = logging.Logger(name)
    handler = logging.StreamHandler()
    h_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(h_format)
    logger.addHandler(handler)
    return logger
