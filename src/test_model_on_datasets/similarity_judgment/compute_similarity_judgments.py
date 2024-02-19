import argparse
import glob
import os
import pathlib
import toml
import inspect
from src.utils.similarity_judgment.generate_report import generate_report

from src.utils.similarity_judgment.run import compute_distance

DEFAULT = {"toml_file": None}


def run_similarity_judgment(toml_file):
    if toml_file is None:
        toml_files = glob.glob(str(pathlib.Path(__file__).parent / "*.toml"))
    else:
        toml_files = [toml_file]
    for toml_f in toml_files:
        with open(toml_f, "r") as f:
            toml_config = toml.load(f)
        compute_distance(**toml_config)
        generate_report(
            os.path.join(
                toml_config["saving_folders"]["results_folder"], "dataframe.csv"
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toml_file",
        "-tomlf",
        default=DEFAULT["toml_file"],
        help="The file containing the parameters of the similarity judgment method. If not specified, it will use all files in this folder.",
    )

    args = parser.parse_known_args()[0]
    run_similarity_judgment(**args.__dict__)
