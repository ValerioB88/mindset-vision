import argparse
import csv
import uuid
from PIL import Image
from pathlib import Path

import toml
import inspect

from src.generate_datasets.visual_illusions.grayscale_shapes.utils import add_arrow

import os
from tqdm import tqdm
import os
import sty
import pathlib
from src.utils.drawing_utils import resize_and_paste

from src.utils.misc import apply_antialiasing, delete_and_recreate_path
import numpy as np

category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS = {
    "antialiasing": False,
    "behaviour_if_present": "overwrite",
    "steps_arrow": 5,
    "canvas_size": [224, 224],
    "output_folder": f"data/{category_folder}/{name_dataset}",
    "grayscale_background": 0,
}


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    grayscale_background=DEFAULTS["grayscale_background"],
    steps_arrow=DEFAULTS["steps_arrow"],
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

    (output_folder / "all_images").mkdir(exist_ok=True, parents=True)

    img_path = Path("assets") / "adelson_checkerboard" / "nochars_nobg.png"
    original_image = Image.open(img_path)
    img = Image.new("RGBA", canvas_size, (*(grayscale_background,) * 3, 0))
    resize_and_paste(original_image, img)
    img = img.convert("L")

    width, height = img.size
    coordinates = [
        (x, y)
        for x in range(steps_arrow, width, steps_arrow)
        for y in range(steps_arrow, height, steps_arrow)
    ]
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Target Pixel Color", "Target Pixel Location", "BackgroundColor"]
        )
        for coordinate in tqdm(coordinates, colour="green"):
            pixel_color = img.getpixel(coordinate)
            img_copy = img.copy()
            img_copy = add_arrow(img_copy, tuple(coordinate))
            image_path = Path("all_images") / f"{uuid.uuid4().hex[:8]}.png"
            img_copy = apply_antialiasing(img_copy) if antialiasing else img_copy
            img_copy.save(output_folder / image_path)

            writer.writerow(
                [str(image_path), pixel_color, coordinate, grayscale_background]
            )


if __name__ == "__main__":
    description = "This dataset simply consists of the Adelson Checker Shadow illusory image replicated many times, grayscaled, with a white arrow systematically placed at different locations of the canvas, covering the whole checkerboard. This dataset is supposed to be used in conjunction with a color-picker decoder, that is a decoder trained on the Grayscale Shape dataset. "
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output_folder",
        "-o",
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure used to create the dataset",
        default=DEFAULTS["output_folder"],
    )
    parser.add_argument(
        "--grayscale_background",
        "-gray_bg",
        help="The background in grayscale value",
        default=DEFAULTS["grayscale_background"],
        type=int,
    )

    parser.add_argument(
        "--steps_arrow",
        "-s",
        help="The arrow will be placed at every s steps.",
        default=DEFAULTS["steps_arrow"],
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default=DEFAULTS["canvas_size"],
        help="The size of the canvas. If called through command line, a string in the format NxM.",
        type=lambda x: (
            tuple([int(i) for i in x.split("x")]) if isinstance(x, str) else x
        ),
    )

    parser.add_argument(
        "--antialiasing",
        "-antial",
        help="Specify whether we want to use antialiasing",
        action="store_true",
        default=DEFAULTS["antialiasing"],
    )
    parser.add_argument(
        "--behaviour_if_present",
        "-if_pres",
        help="What to do if the dataset folder is already present? Choose between [overwrite], [skip]",
        default=DEFAULTS["behaviour_if_present"],
    )
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
