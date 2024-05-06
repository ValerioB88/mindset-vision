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

config_blob_vs_diff_blobs = Path(__file__).parent / "comparison_blob_vs_diff_blob.toml"
config_blob_vs_diff_blob_texturized = (
    Path(__file__).parent / "comparison_blob_vs_diff_blob_texturized.toml"
)
config_blob_vs_same_blob_texturized = (
    Path(__file__).parent / "comparison_blob_vs_same_blob_texturized.toml"
)

gen_dataset_toml_file = Path(__file__).parent / "generate_texturized_blobs.toml"


with open(config_blob_vs_diff_blob_texturized, "r") as f:
    blob_vs_diff_blob_text = toml.load(f)

with open(config_blob_vs_diff_blobs, "r") as f:
    blob_vs_diff_blob = toml.load(f)

with open(config_blob_vs_same_blob_texturized, "r") as f:
    blob_vs_same_blob_text = toml.load(f)


if not os.path.exists(
    Path(blob_vs_diff_blob_text["basic_info"]["annotation_file_path"])
):
    generate_datasets_from_toml_file(gen_dataset_toml_file)

num_blobs = toml.load(str(gen_dataset_toml_file))[
    "shape_and_object_recognition/texturized_blobs"
]["num_blobs"]
original_saving_folder = blob_vs_diff_blob["saving_folders"]["results_folder"]

print("Blobs vs Other Blobs")
for i in range(num_blobs):
    print(f"Reference Blob: {i}")
    blob_vs_diff_blob["basic_info"]["reference_level"] = i
    blob_vs_diff_blob["saving_folders"]["results_folder"] = (
        original_saving_folder + f"_{i}"
    )

    compute_distance(**blob_vs_diff_blob)


compute_distance(**blob_vs_same_blob_text)
compute_distance(**blob_vs_diff_blob_text)
