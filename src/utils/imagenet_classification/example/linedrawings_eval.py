import os
import pandas as pd
import toml
import inspect

from src.utils.dataset_utils import ImageNetClasses
from src.utils.imagenet_classification.eval import classification_evaluate
import pathlib
from pathlib import Path
from src.generate_datasets_from_toml import generate_datasets_from_toml_file


from src.generate_datasets.shape_and_object_recognition.linedrawings.generate_dataset import (
    generate_all as linedrawings_generate,
)


gen_dataset_toml_file = Path(__file__).parent / "generate_linedrawings_dataset.toml"
classification_toml_file = Path(__file__).parent / "linedrawings_config.toml"

with open(gen_dataset_toml_file, "r") as f:
    gen_dataset_config = toml.load(f)


with open(classification_toml_file, "r") as f:
    classification_config = toml.load(f)

annot_file_path = (
    Path(
        gen_dataset_config["shape_and_object_recognition/linedrawings"]["output_folder"]
    )
    / "annotation.csv"
)
## If the annotation file doesn't exist, it means the dataset doesn't exist, and we create it.
if not os.path.exists(Path(annot_file_path)):
    generate_datasets_from_toml_file(gen_dataset_toml_file)

# We need to add the ImageNetClassIndex column, based on the "Class" column.
add_class = ImageNetClasses()
add_class.add_to_annotation_file_path(
    annot_file_path,
    "Class",
    str(Path(annot_file_path).parent / "annotation_w_imagenet_idxs.csv"),
)
classification_evaluate(**classification_config)
