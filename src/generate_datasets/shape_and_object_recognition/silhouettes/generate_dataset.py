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


class DrawLinedrawings(DrawStimuli):
    def __init__(self, obj_longest_side, input_image_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.input_image_type = input_image_type

    def get_linedrawings(
        self,
        image_path,
    ):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.input_image_type == "linedrawings":
            contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            mask = np.ones_like(img) * 255

            cv2.drawContours(mask, contours, -1, (0), thickness=cv2.FILLED)

            [
                cv2.drawContours(mask, [c], -1, (0), thickness=cv2.FILLED)
                for c in contours
            ]
        else:
            mask = cv2.bitwise_not(binary_img)
        mask = ImageOps.invert(Image.fromarray(mask).convert("L"))

        canvas = paste_linedrawing_onto_canvas(mask, self.create_canvas(), self.fill)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 200,
        "image_input_folder": "assets/baker_2018_linedrawings/cropped/",
        "output_folder": f"data/{category_folder}/{name_dataset}_from_linedrawings_baker_2018",
        "antialiasing": False,
        "input_image_type": "linedrawings",
    }
)

DEFAULTS_bis = DEFAULTS.copy()
DEFAULTS_bis["input_image_type"] = "silhouettes"
DEFAULTS_bis["image_input_folder"] = "assets/baker_2022_silhouettes/cropped"
DEFAULTS_bis["output_folder"] = (
    f"data/{category_folder}/{name_dataset}_from_silhouettes_baker_2022"
)

DEFAULTS = [DEFAULTS, DEFAULTS_bis]


def generate_all(
    object_longest_side=DEFAULTS[0]["object_longest_side"],
    image_input_folder=DEFAULTS[0]["image_input_folder"],
    output_folder=DEFAULTS[0]["output_folder"],
    canvas_size=DEFAULTS[0]["canvas_size"],
    background_color=DEFAULTS[0]["background_color"],
    antialiasing=DEFAULTS[0]["antialiasing"],
    behaviour_if_present=DEFAULTS[0]["behaviour_if_present"],
    input_image_type=DEFAULTS[0]["input_image_type"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}
    image_input_folder = Path(image_input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)

    all_categories = [i.stem for i in image_input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    ds = DrawLinedrawings(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        input_image_type=input_image_type,
    )
    jpg_files = list(image_input_folder.rglob("*.jpg"))
    png_files = list(image_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            image_name = img_path.stem
            img = ds.get_linedrawings(img_path)
            unique_hex = uuid.uuid4().hex[:8]

            path = Path(class_name) / f"{image_name}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, class_name, ds.background, n])

    return str(output_folder)


if __name__ == "__main__":
    description = "We generate a dataset of silhouettes by using samples from Baker & Elder (2022) (9 classes from ImageNet, each class containing 40 samples). The user can specify any folder containing silhouettes. Alternatively, the user can also specify a folder containing line-drawings (black strokes on a white background), which will be converted into silhouettes.\nREF: Baker, Nicholas, and James H. Elder. 'Deep Learning Models Fail to Capture the Configural Nature of Human Shape Perception'. iScience 25, no. 9 (16 September 2022): 104913. https://doi.org/10.1016/J.ISCI.2022.104913."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS[0]["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS[0]["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS[0]["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )
    parser.add_argument(
        "--input_image_type",
        "-img_t",
        default=DEFAULTS[0]["input_image_type"],
        help="Either [silhouettes] or [linedrawings]. Default is [linedrawings]. Both are supposed to be blackstroke over a black canvas. If using linedrawings, they will first be converted into silhouettes.",
    )
    parser.add_argument(
        "--image_input_folder",
        "-fld",
        dest="image_input_folder",
        help="A folder containing the image input types (linedrawings/silhouettes). We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS[0]["image_input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
