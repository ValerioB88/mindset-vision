import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import sty
import PIL.Image as Image
import toml
import inspect

from src.utils.drawing_utils import (
    DrawStimuli,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os

DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm

from PIL import ImageOps


class DrawLinedrawings(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        canvas = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 200,
        "input_folder": "assets/kubilius_2017/pngs",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    object_longest_side=DEFAULTS["object_longest_side"],
    input_folder=DEFAULTS["input_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    all_categories = [i.stem for i in input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    ds = DrawLinedrawings(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )
    jpg_files = list(input_folder.rglob("*.jpg"))
    png_files = list(input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            img = ds.get_linedrawings(img_path)
            path = Path(class_name) / f"{n}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])
    return str(output_folder)


if __name__ == "__main__":
    description = "2D line segments recreated through vector graphics based on Kubilius et al. 2017. A feature dimension (such as the curvature of a Geon) is altered from a singular value (e.g. straight contour with 0 curvature) to two different values (e.g. slightly curved or very curved). The `reference` condition is the item with the intermediate value; the `MP change` condition consists of the sample with a greater non-singular value (that is, from slight curvature to greater curvature, which corresponds to a MP change), and the `NAP` change condition includes the samples with the singular value (that is, from slight curvature to non curvature, which corresponds to a NAP change). \nREF: Kubilius, Jonas, Charlotte Sleurs, and Johan Wagemans. 'Sensitivity to Nonaccidental Configurations of Two-Line Stimuli'. I-Perception 8, no. 2 (1 April 2017): 1-12. https://doi.org/10.1177/2041669517699628/"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--input_folder",
        "-fld",
        dest="input_folder",
        help="A folder containing the NAP vs MP 2D lines stimuli.",
        default=DEFAULTS["input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
