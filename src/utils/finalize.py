"""
This file will perform all the operations that needs to be done when all code is written: 
- generate the toml configs (for both full and lite version), 
- generate the datasets (full and lite), 
- and publish the datasets on kaggle.
"""

import glob
from src.generate_datasets_from_toml import generate_datasets_from_toml_file
import src.utils.generate_default_pars_toml_file
from src.utils.misc import modify_toml
from src.utils.publish_kaggle import publish


def finalize(generate_tomls, generate_datasets, publish_kaggle):
    toml_all_full = "generate_all_datasets.toml"
    toml_all_lite = "generate_all_datasets_lite.toml"

    if generate_tomls:
        print("generating TOMLs")
        src.utils.generate_default_pars_toml_file.create_config(toml_all_full)
        src.utils.generate_default_pars_toml_file.generate_lite(
            toml_all_full, toml_all_lite
        )

    if generate_datasets:
        print("generating datasets")
        generate_datasets_from_toml_file(toml_all_full)
        generate_datasets_from_toml_file(toml_all_lite)

    if publish_kaggle:
        print("publish on Kaggle")
        publish(data_type="full")
        publish(data_type="lite")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finalize dataset creation/publication"
    )
    parser.add_argument(
        "--generate_tomls", "-t", action="store_true", help="Generate toml files"
    )
    parser.add_argument(
        "--generate_datasets", "-d", action="store_true", help="Generate datasets"
    )
    parser.add_argument(
        "--publish_kaggle", "-k", action="store_true", help="Publish datasets on kaggle"
    )
    args = parser.parse_args()
    finalize(args.generate_tomls, args.generate_datasets, args.publish_kaggle)
