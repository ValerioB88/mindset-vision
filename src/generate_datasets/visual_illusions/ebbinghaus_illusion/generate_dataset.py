import csv
import os
import pathlib

import numpy as np
import argparse

import sty
import toml
import inspect

from tqdm import tqdm
import uuid

from .utils import (
    DrawEbbinghaus,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS["num_samples_scrambled"] = 5000
DEFAULTS["num_samples_illusory"] = 50

DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    num_samples_illusory=DEFAULTS["num_samples_illusory"],
    num_samples_scrambled=DEFAULTS["num_samples_scrambled"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [
        (output_folder / i).mkdir(parents=True, exist_ok=True)
        for i in ["scrambled_circles", "small_flankers", "big_flankers"]
    ]

    ds = DrawEbbinghaus(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Category",
                "NormSizeCenterCircle",
                "NormSizeOtherFlankers",
                "NumFlankers",
                "BackgroundColor",
                "Shift",
            ]
        )
        for i in tqdm(range(num_samples_scrambled)):
            r_c = np.random.uniform(0.1, 0.4)
            img = ds.create_random_ebbinghaus(
                r_c=r_c,
                n=5,
                flankers_size_range=(0.04, 0.3),
                colour_center_circle=(255, 0, 0),
            )
            path = pathlib.Path("scrambled_circles") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow([path, "scrambled_circles", r_c, "", 5, ds.background, ""])
        for i in tqdm(range(num_samples_illusory)):
            number_flankers = 5
            r_c = np.random.uniform(0.1, 0.4)
            r2 = np.random.uniform(0.24, 0.3)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(
                r_c=r_c,
                d=0.02 + (r_c + r2) / 2,
                r2=r2,
                n=number_flankers,
                shift=shift,
                colour_center_circle=(255, 0, 0),
            )
            path = pathlib.Path("big_flankers") / f"{r_c:.5f}_{i}.png"
            img.save(output_folder / path)
            writer.writerow(
                [path, "big_flankers", r_c, r2, number_flankers, ds.background, shift]
            )

            number_flankers = 8
            r_c = np.random.uniform(0.1, 0.4)
            r2 = np.random.uniform(0.04, 0.15)
            shift = np.random.uniform(0, np.pi)
            img = ds.create_ebbinghaus(
                r_c=r_c,
                d=0.02 + (r_c + r2) / 2,
                r2=r2,
                n=number_flankers,
                shift=shift,
                colour_center_circle=(255, 0, 0),
            )
            unique_hex = uuid.uuid4().hex[:8]
            path = pathlib.Path("small_flankers") / f"{r_c:.5f}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow(
                [path, "small_flankers", r_c, r2, number_flankers, ds.background, shift]
            )
    return str(output_folder)


if __name__ == "__main__":
    description = "A red target circle is surrounded by a fixed number of white circles (flankers) on a uniform background. In the two illusory conditions (`big' and `small' flankers) the flankers are disposed around the target circle, and they all have the same size within each sample. In the `scrambled' condition the target circle is still placed in the center, but white circles with random sizes are randomly placed on the canvas. Across illusory samples, we varied the radii of the flankers, the radius of the target circle, the displacement of the flankers around the target. "
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_scrambled",
        "-nss",
        type=int,
        default=DEFAULTS["num_samples_scrambled"],
        help="How many samples to generated for the scrambled up conditions",
    )

    parser.add_argument(
        "--num_samples_illusory",
        "-nsi",
        type=int,
        default=DEFAULTS["num_samples_illusory"],
        help="How many samples to generated for the illusory conditions (small and big flankers)",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
