import os
import pandas as pd
import toml
import inspect

from src.utils.decoder.eval import decoder_evaluate
import pathlib


dataset_folder = (
    pathlib.Path("examples") / "data" / "visual_illusion" / "ebbinghaus_illusion"
)

# We create a dataset just for this example.
from src.generate_datasets.visual_illusions.ebbinghaus_illusion.generate_dataset import (
    generate_all as ebbinghaus_illusion_generate,
)

# Notice: below is how you create datasets in script, specifying the arguments in code. However, you could also create dataset with
# python -m src.generate_datasets_from_toml src/utils/decoder/examples/regression/ebbinghaus/generate_ebbinghaus_illusion.toml - in which case the config parameters used in the toml file will be used.
if not pathlib.Path(dataset_folder).exists():
    ebbinghaus_illusion_generate(
        output_folder=dataset_folder,
        num_samples=1000,
    )


with open(os.path.dirname(__file__) + "/eval.toml", "r") as f:
    toml_config = toml.load(f)
decoder_evaluate(**toml_config)

##
