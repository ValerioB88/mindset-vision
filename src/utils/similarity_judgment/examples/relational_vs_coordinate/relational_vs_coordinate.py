import argparse
import glob
import pathlib
from anyio import Path
import toml
import inspect
import os
from src.generate_datasets_from_toml import generate_datasets_from_toml_file
from src.utils.similarity_judgment.generate_report import generate_report
from src.utils.similarity_judgment.run import compute_distance

sim_judgment_toml_file = (
    Path(__file__).parent / "sim_judgm_relational_vs_coordinate.toml"
)
gen_dataset_toml_file = (
    Path(__file__).parent / "generate_relational_vs_coordinate_dataset.toml"
)


with open(sim_judgment_toml_file, "r") as f:
    sim_judgm_toml = toml.load(f)

## If the annotation file doesn't exist, it means the dataset doesn't exist, and we create it.
if not os.path.exists(Path(sim_judgm_toml["basic_info"]["annotation_file_path"])):
    generate_datasets_from_toml_file(gen_dataset_toml_file)

# Notice that you could pass the parameters without using a toml file! Useful if you need to change the parameters programmatically and don't want to generate a new toml file everytime.
compute_distance(**sim_judgm_toml)
generate_report(
    os.path.join(sim_judgm_toml["saving_folders"]["results_folder"], "dataframe.csv")
)
