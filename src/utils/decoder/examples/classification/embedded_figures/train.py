import os
import pandas as pd
import toml
import inspect

from src.utils.decoder.train import decoder_train
import pathlib


dataset_folder = (
    pathlib.Path("examples")
    / "data"
    / "shape_and_object_recognition"
    / "embedded_figures"
)

# We create a  dataset just for this example.
from src.generate_datasets.shape_and_object_recognition.embedded_figures.generate_dataset import (
    generate_all as embedded_figures_generate,
)

# Notice: below is how you create datasets in script, specifying the arguments in code. However, you could also create dataset with
# python -m src.generate_datasets_from_toml src/utils/decoder/examples/classification/embedded_figures/generate_embedded_figures.toml - in which case the config parameters used in the toml file will be used.
if not pathlib.Path(dataset_folder).exists():
    embedded_figures_generate(
        output_folder=dataset_folder,
        num_samples=10000,
    )


with open(os.path.dirname(__file__) + "/train.toml", "r") as f:
    toml_config = toml.load(f)
decoder_train(**toml_config)
