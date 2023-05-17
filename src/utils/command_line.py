from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
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
