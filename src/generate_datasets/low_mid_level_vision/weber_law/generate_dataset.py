import argparse
import csv
import os
import pathlib
from tkinter import N
import numpy as np
import uuid
import toml
import inspect

from torchvision.transforms import transforms, InterpolationMode
import sty
from torch import rand
from PIL import Image, ImageDraw
from tqdm import tqdm
from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import add_general_args, delete_and_recreate_path
from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def draw_line(length, width_range, lum_range, len_var, len_unit, size_imx, size_imy):
    im = Image.new("RGB", (size_imx, size_imy), color="black")

    ### Randomly draw width and luminance from range
    width = np.random.randint(width_range[0], width_range[1])
    lum = np.random.randint(lum_range[0], lum_range[1])
    delta_len = np.random.randint(len_var[0], len_var[1])
    len_in_pix = length * len_unit
    new_len_in_pix = len_in_pix + delta_len

    ### Find coordinates of line in the middle of image
    xc = size_imx / 2
    yc = size_imy / 2
    x0, y0 = xc - (new_len_in_pix / 2), yc
    x1, y1 = xc + (new_len_in_pix / 2), yc
    bbox = [(x0, y0), (x1, y1)]

    drawing = ImageDraw.Draw(im)
    drawing.line(bbox, width=width, fill=(lum, lum, lum))

    return im


class DrawWeberLength(DrawStimuli):
    def gen_stim(
        self,
        length,
        width,
        lum,
    ):
        img = self.create_canvas()
        x0, y0 = self.canvas_size[0] / 2 - (length / 2), self.canvas_size[1] / 2
        x1, y1 = self.canvas_size[0] / 2 + (length / 2), self.canvas_size[1] / 2
        bbox = [(x0, y0), (x1, y1)]

        drawing = ImageDraw.Draw(img)
        drawing.line(bbox, width=width, fill=(lum, lum, lum))
        return img


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS["num_samples_per_condition"] = 50
DEFAULTS["max_line_length"] = 50
DEFAULTS["min_line_length"] = 5
DEFAULTS["interval_line_length"] = 1
DEFAULTS["min_grayscale"] = 50
DEFAULTS["max_grayscale"] = 255
DEFAULTS["interval_grayscale"] = 20
DEFAULTS["width"] = 2
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    num_samples_per_condition=DEFAULTS["num_samples_per_condition"],
    max_line_length=DEFAULTS["max_line_length"],
    min_line_length=DEFAULTS["min_line_length"],
    interval_line_length=DEFAULTS["interval_line_length"],
    min_grayscale=DEFAULTS["min_grayscale"],
    max_grayscale=DEFAULTS["max_grayscale"],
    interval_grayscale=DEFAULTS["interval_grayscale"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    width=DEFAULTS["width"],
) -> str:
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    lengths_conditions = range(min_line_length, max_line_length, interval_line_length)
    grayscale_conditions = range(min_grayscale, max_grayscale, interval_grayscale)
    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    (output_folder / "all_images").mkdir(parents=True, exist_ok=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "BackgroundColor", "Length", "Width", "Luminance", "IterNum"]
        )
        ds = DrawWeberLength(
            background=background_color,
            canvas_size=canvas_size,
            antialiasing=antialiasing,
        )
        if not isinstance(width, str):
            num_samples_per_condition = 1

        for len in lengths_conditions:
            for gr in grayscale_conditions:
                for n in range(num_samples_per_condition):
                    # with a canvas_size_y of 224, this is from 1 to 5 pixels width.
                    w = (
                        int(np.random.uniform(0.0044, 0.02232) * ds.canvas_size[1])
                        if width == "rnd"
                        else width
                    )
                    # lum = int(np.random.uniform(100, 256))
                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = f"{len}_{gr}_{unique_hex}.png"
                    img = ds.gen_stim(len, w, gr)
                    img.save(output_folder / "all_images" / img_path)

                ## TODO: Notice that we do not apply object transformations. In MindSet, we assume that image transformations are all done somewhere else, e.g. during training, by pycharm (there are some exception to this rule, in those cases where pycharm would screw up the image background). However, here Gaurav found particularly important to separate train/testing by translation location. This is tricky. If I do perform transformation here (and save the location in the annotation file, which then the user can use to separate different transformation for train/test), why not on ALL other datasets? We need to think about this.
                writer.writerow([img_path, ds.background, len, w, gr, n])

    return str(output_folder)


if __name__ == "__main__":
    description = "A simple horizontal white line with varying length and brightness. Configurable parameters include line width, min/max length/brightnes"
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_per_condition",
        "-ns",
        default=DEFAULTS["num_samples_per_condition"],
        help="The number of samples to generate for each length and brightness condition. This will vary the width, if width is [rnd]. If width is a specific value, num_samples_per_condition will be set to 1",
    )
    parser.add_argument(
        "--max_line_length",
        "-maxll",
        default=DEFAULTS["max_line_length"],
        help="The maximum line length (in pixels) to use",
    )

    parser.add_argument(
        "--min_line_length",
        "-minll",
        default=DEFAULTS["min_line_length"],
        help="The minimum line length (in pixels) to use",
    )

    parser.add_argument(
        "--interval_line_length",
        "-ill",
        default=DEFAULTS["interval_line_length"],
        help="The Interval line length to use",
    )
    parser.add_argument(
        "--width",
        "-w",
        default=DEFAULTS["width"],
        help="Width of the line. [rnd] to use a random width, otherwise specify a value (in pixel).",
    )
    parser.add_argument(
        "--max_grayscale",
        "-maxgr",
        default=DEFAULTS["max_grayscale"],
        help="The maximum grayscale value to use for the brightness condition",
    )

    parser.add_argument(
        "--min_grayscale",
        "-mingr",
        default=DEFAULTS["min_grayscale"],
        help="The minumum grayscale value to use for the brightness condition",
    )

    parser.add_argument(
        "--interval_grayscale",
        "-igr",
        default=DEFAULTS["interval_grayscale"],
        help="The Interval grayscale value to use",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
