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
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)
from PIL import Image, ImageDraw, ImageChops, ImageOps


from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import os

DEFAULTS = BASE_DEFAULTS.copy()
from tqdm import tqdm

import cv2
from PIL import Image


class DrawLinedrawings(DrawStimuli):
    def __init__(self, obj_longest_side, convert_to_silhouettes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.convert_to_silhouettes = convert_to_silhouettes

    def get_linedrawings(self, image_path, type):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        if self.convert_to_silhouettes:
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

        silhouette = Image.fromarray(mask)
        width, height = silhouette.size
        top_half = silhouette.crop((0, 0, width, height // 2))
        bottom_half = silhouette.crop((0, height // 2, width, height))

        if type in ["frankenstein", "fragmented"]:
            top_half = top_half.transpose(Image.FLIP_LEFT_RIGHT)

        top_half_np = np.array(top_half)
        bottom_half_np = np.array(bottom_half)
        if type == "frankenstein":
            # Take the first black pixel of the bottom row ot the top part
            top = np.min(np.where(top_half_np[-1] == 0))
            # and the first black pixel of the top row of the bottom part
            bottom = np.min(np.where(bottom_half_np[0] == 0))
        elif type == "fragmented":
            # take the first black pixel of the bottom row of the top part
            top = np.min(np.where(top_half_np[-1] == 0))
            # and the LAST black pixel of the top row of the bottom part
            bottom = np.max(np.where(bottom_half_np[0] == 0))
        else:  # type == "whole":
            bottom = 0
            top = 0
        # Calculate the offset based on the leftmost black pixels
        top_offset = max(0, bottom - top)
        bottom_offset = max(0, top - bottom)

        # Make the new canvas larger to accomodate for weird splitting
        new_canvas = Image.fromarray(
            np.ones(
                (
                    self.canvas_size[0],
                    max(top_half.size[0], bottom_half.size[0])
                    + max(top_offset, bottom_offset),
                )
            )
            * 255
        ).convert("RGB")

        new_canvas.paste(
            top_half,
            (
                top_offset,  # new_canvas.size[0] // 2 - top_cropped.size[1] // 2 + offset,
                new_canvas.size[1] // 2 - top_half.size[1],
            ),
        )
        new_canvas.paste(
            bottom_half,
            (
                bottom_offset,  # new_canvas.size[0] // 2 - top_cropped.size[1] // 2,
                new_canvas.size[1] // 2,
            ),  # + bottom_half.size[1] // 2)
        )

        cs = tuple(
            np.array(self.canvas_size)
            * max(np.array(new_canvas.size) / self.canvas_size)
        )
        canvas = self.create_canvas(size=[int(i) for i in cs])
        canvas.paste(
            ImageOps.invert(new_canvas.convert("L")),
            (
                canvas.size[0] // 2 - new_canvas.size[0] // 2,
                canvas.size[1] // 2 - new_canvas.size[1] // 2,
            ),
        )
        canvas = canvas.resize(self.canvas_size)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 120,
        "image_input_folder": "assets/baker_2018_linedrawings/cropped/",
        "output_folder": f"data/{category_folder}/{name_dataset}_from_linedrawings_baker_2018",
        "antialiasing": False,
        "convert_to_silhouettes": 0,
    }
)

DEFAULTS_bis = DEFAULTS.copy()
DEFAULTS_bis["image_input_folder"] = "assets/baker_2022_silhouettes/cropped"
DEFAULTS_bis["output_folder"] = (
    f"data/{category_folder}/{name_dataset}_from_silhouettes_baker_2022"
)
DEFAULTS_bis["convert_to_silhouettes"] = 0

DEFAULTS_tris = DEFAULTS.copy()
DEFAULTS_tris["convert_to_silhouettes"] = 1
DEFAULTS_tris["output_folder"] = (
    f"data/{category_folder}/{name_dataset}_silhouettes_from_linedrawings_baker_2018"
)


DEFAULTS = [DEFAULTS, DEFAULTS_bis, DEFAULTS_tris]

def_index = 1


def generate_all(
    object_longest_side=DEFAULTS[def_index]["object_longest_side"],
    image_input_folder=DEFAULTS[def_index]["image_input_folder"],
    output_folder=DEFAULTS[def_index]["output_folder"],
    canvas_size=DEFAULTS[def_index]["canvas_size"],
    background_color=DEFAULTS[def_index]["background_color"],
    antialiasing=DEFAULTS[def_index]["antialiasing"],
    behaviour_if_present=DEFAULTS[def_index]["behaviour_if_present"],
    convert_to_silhouettes=DEFAULTS[def_index]["convert_to_silhouettes"],
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
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    all_categories = [i.stem for i in image_input_folder.glob("*")]

    ds = DrawLinedrawings(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        convert_to_silhouettes=convert_to_silhouettes,
    )
    conditions = ["whole", "fragmented", "frankenstein"]
    [
        [
            (output_folder / c / cat).mkdir(exist_ok=True, parents=True)
            for cat in all_categories
        ]
        for c in conditions
    ]
    jpg_files = list(image_input_folder.rglob("*.jpg"))
    png_files = list(image_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Class", "BackgroundColor", "IterNum"])
        for n, img_path in enumerate(tqdm(image_files)):
            for t in conditions:
                class_name = img_path.parent.stem
                image_name = img_path.stem
                img = ds.get_linedrawings(img_path, type=t)
                path = Path(t) / class_name / f"{image_name}.png"
                img.save(output_folder / path)
                writer.writerow([path, class_name, ds.background, n])
    return str(output_folder)


if __name__ == "__main__":
    description = "This dataset is inspired by Baker and Elder (2022). Instead of replicating their dataset , we wrote a script to automatically generate fragmented and `Frankenstein' versions of a silhouette or a line drawing. The user can specify their own line drawing or silhouette folder, to generate a different variety of fragmented or Frankenstein images. We also provide a simple replication of Baker et al. 2022 stimuli in ../global_change_baker2022. \nREF: Baker, Nicholas, and James H. Elder. 'Deep Learning Models Fail to Capture the Configural Nature of Human Shape Perception'. iScience 25, no. 9 (16 September 2022). https://doi.org/10.1016/J.ISCI.2022.104913."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS[def_index]["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS[def_index]["antialiasing"])
    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS[def_index]["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )

    parser.add_argument(
        "--image_input_folder",
        "-fld",
        dest="image_input_folder",
        help="A folder containing the image input types (linedrawings/silhouettes). We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS[def_index]["image_input_folder"],
    )
    parser.add_argument(
        "--convert_to_silhouettes",
        dest="convert_to_silhouettes",
        default=DEFAULTS[def_index]["convert_to_silhouettes"],
        type=lambda x: bool(str(x)),
        help="Use 1 to convert to silhouettes, 0 to not convert. Set to 0 if the images are already silhouettes!",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
