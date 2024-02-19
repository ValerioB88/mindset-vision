import argparse
import csv
from pathlib import Path
import uuid
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
from PIL import ImageOps, Image


class DrawDottedImage(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def dotted_image(self, image_path, dot_distance, dot_size):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)

        _, binary_img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        contours, b = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dotted_img = np.ones_like(img) * 255

        def draw_dot(image, x, y, size, color):
            half_size = size // 2
            cv2.rectangle(
                image,
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                color,
                -1,
            )

        for contour in contours:
            for i, point in enumerate(contour):
                if i % dot_distance == 0:
                    x, y = point[0]
                    draw_dot(dotted_img, x, y, dot_size, color=0)

        dotted_img = Image.fromarray(dotted_img)
        dotted_img = ImageOps.invert(dotted_img.convert("L"))

        canvas = paste_linedrawing_onto_canvas(
            dotted_img, self.create_canvas(), self.fill
        )

        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "object_longest_side": 200,
        "linedrawing_input_folder": "assets/baker_2018_linedrawings/cropped/",
        "dot_distance": 5,
        "dot_size": 1,
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
    }
)


def generate_all(
    object_longest_side=DEFAULTS["object_longest_side"],
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    dot_distance=DEFAULTS["dot_distance"],
    dot_size=DEFAULTS["dot_size"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    linedrawing_input_folder = Path(linedrawing_input_folder)
    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    all_categories = [i.stem for i in linedrawing_input_folder.glob("*")]

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]

    ds = DrawDottedImage(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
    )
    jpg_files = list(linedrawing_input_folder.rglob("*.jpg"))
    png_files = list(linedrawing_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Class", "BackgroundColor", "DotDistance", "DotSize", "IterNum"]
        )
        for n, img_path in enumerate(tqdm(image_files)):
            class_name = img_path.parent.stem
            name_sample = img_path.stem

            img = ds.dotted_image(
                img_path, dot_distance=dot_distance, dot_size=dot_size
            )

            path = Path(class_name) / f"{name_sample}.png"
            img.save(output_folder / path)
            writer.writerow(
                [path, class_name, ds.background, dot_distance, dot_size, n]
            )
    return str(output_folder)


if __name__ == "__main__":
    description = "This dataset consists of modification of line drawings (by default, we use line-drawings from Baker et al. (2018). The line drawings are 'dottified'. The user can specify the dots size and the distance between each dot.\nREF: Baker, Nicholas, Hongjing Lu, Gennady Erlikhman, and Philip J. Kellman. 'Deep Convolutional Networks Do Not Classify Based on Global Object Shape'. PLoS Computational Biology 14, no. 12 (2018): 1-43. https://doi.org/10.1371/journal.pcbi.1006613."
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
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )

    parser.add_argument(
        "--dot_distance",
        "-dd",
        default=DEFAULTS["dot_distance"],
        help="Distance between dots",
        type=int,
    )
    parser.add_argument(
        "--dot_size",
        "-ds",
        default=DEFAULTS["dot_size"],
        help="Size of each dot",
        type=int,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
