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
    get_mask_from_linedrawing,
    paste_linedrawing_onto_canvas,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)
import uuid
from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os

DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm

import cv2
from PIL import Image
from PIL import ImageOps


class DrawBakerStimuli(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_linedrawings(
        self,
        image_path,
    ):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.bitwise_not(binary_img)
        mask = ImageOps.invert(Image.fromarray(mask).convert("L"))

        canvas = paste_linedrawing_onto_canvas(mask, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 200,
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    object_longest_side=DEFAULTS["object_longest_side"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}
    image_input_folder = Path("assets/baker_2022")
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    all_categories = [i.stem for i in image_input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    ds = DrawBakerStimuli(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )
    jpg_files = list(image_input_folder.rglob("*.jpg"))
    png_files = list(image_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            type_image = img_path.parent.parts[-2]
            image_name = img_path.stem
            img = ds.get_linedrawings(img_path)
            unique_hex = uuid.uuid4().hex[:8]

            path = Path(type_image) / class_name / f"{image_name}_{unique_hex}.png"
            (output_folder / path).parent.mkdir(exist_ok=True, parents=True)
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)


if __name__ == "__main__":
    description = "This is a simple recreationg (with additional configuration parameters) of the stimuli presented in Baker et al. 2022, including 9 classes from ImageNet, each containing 40 samples. We also offer a script designed to apply a similar transformation to any line drawings or silhouette samples (see ../global_change). \nREF: Baker, Nicholas, and James H. Elder. 'Deep Learning Models Fail to Capture the Configural Nature of Human Shape Perception'. iScience 25, no. 9 (16 September 2022). https://doi.org/10.1016/J.ISCI.2022.104913."
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
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
