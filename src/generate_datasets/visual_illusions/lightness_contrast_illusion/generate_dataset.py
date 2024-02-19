import argparse
import csv
import uuid
from PIL import Image
from pathlib import Path

import toml
import inspect

from src.generate_datasets.visual_illusions.grayscale_shapes.utils import add_arrow
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import os
import sty
import pathlib
from src.utils.drawing_utils import DrawStimuli

from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)
import numpy as np
from src.utils.misc import DEFAULTS as BASE_DEFAULTS

category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS = {
    "canvas_size": [224, 224],
    "antialiasing": False,
    "behaviour_if_present": "overwrite",
    "steps_arrow": 30,
    "square_color": 200,
    "steps_bg_color": 20,
    "output_folder": f"data/{category_folder}/{name_dataset}",
}


def generate_all(
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    steps_bg_color=DEFAULTS["steps_bg_color"],
    steps_arrow=DEFAULTS["steps_arrow"],
    square_color=DEFAULTS["square_color"],
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

    coordinates = [
        (x, y)
        for x in range(steps_arrow, canvas_size[0], steps_arrow)
        for y in range(steps_arrow, canvas_size[1], steps_arrow)
    ]
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Target Pixel Color",
                "Target Pixel Location",
                "Background Color",
                "Rectangle Color",
            ]
        )
        for coordinate in tqdm(coordinates, colour="green"):
            grayscale_background_all = np.arange(0, 255, steps_bg_color)
            for background_c in grayscale_background_all:
                image = Image.new("RGBA", canvas_size, (background_c,) * 4)

                draw = ImageDraw.Draw(image)
                width, height = image.size
                size_rect = 80
                draw.rectangle(
                    (
                        width // 2 - size_rect // 2,
                        height // 2 - size_rect // 2,
                        width // 2 + size_rect // 2,
                        height // 2 + size_rect // 2,
                    ),
                    fill=(square_color,) * 3,
                )
                image = add_arrow(image, coordinate)
                image_path = Path("all_images") / f"{uuid.uuid4().hex[:8]}.png"
                image = apply_antialiasing(image) if antialiasing else image
                image = image.convert("L")
                pixel_color = image.getpixel(coordinate)
                image.save(output_folder / image_path)

                writer.writerow(
                    [
                        str(image_path),
                        pixel_color,
                        coordinate,
                        background_c,
                        square_color,
                    ]
                )


if __name__ == "__main__":
    description = "The dataset consists of the standard Lightness Contrast configuration of square within a uniform canvas, having two different grayscale values. The user can specify the grayscale value of the center square, which is kept fixed, while the value of the background is varied. Importantly, each sample is replicated many times with the white arrow marker placed at different location in the canvas. This simple configuration is modified so that it is easy to use in conjunction with a color-picker decoder, that is a decoder trained on the Grayscale Shape dataset. "
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output_folder",
        "-o",
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure used to create the dataset",
        default=DEFAULTS["output_folder"],
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default=DEFAULTS["canvas_size"],
        help="The size of the canvas. If called through command line, a string in the format NxM eg `224x224`.",
        type=lambda x: (
            tuple([int(i) for i in x.split("x")]) if isinstance(x, str) else x
        ),
    )

    parser.add_argument(
        "--antialiasing",
        "-antial",
        dest="antialiasing",
        help="Specify whether we want to enable antialiasing",
        action="store_true",
        default=DEFAULTS["antialiasing"],
    )

    parser.add_argument(
        "--behaviour_if_present",
        "-if_pres",
        help="What to do if the dataset folder is already present? Choose between [overwrite], [skip]",
        default=DEFAULTS["behaviour_if_present"],
    )

    parser.add_argument(
        "--steps_bg_color",
        "-s_bg",
        default=DEFAULTS["steps_bg_color"],
        type=int,
        help="It will generate items which background varies from 0 to 255 in steps specified by this parameter.",
    )

    parser.add_argument(
        "--square_color",
        "-rectcl",
        help="The color of the center square in grayscale int",
        default=DEFAULTS["square_color"],
        type=int,
    )

    parser.add_argument(
        "--steps_arrow",
        "-s_ar",
        help="The arrow will be placed at every s steps.",
        default=DEFAULTS["steps_arrow"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
