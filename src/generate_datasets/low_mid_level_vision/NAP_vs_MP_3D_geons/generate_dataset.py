import argparse
import colorsys
import csv
import inspect

import numpy as np
import sty
import toml
import inspect

from tqdm import tqdm
import cv2
import uuid
import pathlib
import PIL.Image as Image
import os

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

DEFAULTS = BASE_DEFAULTS.copy()
DEFAULTS["stroke_color"] = ""
category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}_standard"
DEFAULTS["shape_folder"] = "assets/amir_geons/cropped/NAPvsMP"
DEFAULTS["object_longest_side"] = 200

DEFAULTS_bis = DEFAULTS.copy()
DEFAULTS_bis["output_folder"] = f"data/{category_folder}/{name_dataset}_no_shades"
DEFAULTS_bis["shape_folder"] = "assets/amir_geons/cropped/NAPvsMP_no_shades"

DEFAULTS_tris = DEFAULTS.copy()
DEFAULTS_tris["output_folder"] = f"data/{category_folder}/{name_dataset}_silhouettes"
DEFAULTS_tris["shape_folder"] = "assets/amir_geons/cropped/NAPvsMP_silhouettes"

DEFAULTS = [DEFAULTS, DEFAULTS_bis, DEFAULTS_tris]


class DrawShape(DrawStimuli):
    def __init__(self, obj_longest_side, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def process_image(self, image_path, shape_color_rgb):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        img = resize_image_keep_aspect_ratio(img, self.obj_longest_side)
        img = paste_linedrawing_onto_canvas(
            Image.fromarray(img), self.create_canvas(), (255, 255, 255)
        )

        new_img = self.create_canvas(size=img.size)
        colors = {(0, 0, 0): self.background}
        if shape_color_rgb:
            shape_color_hls = colorsys.rgb_to_hls(
                *tuple(np.array(shape_color_rgb) // 255)
            )

        data = img.load()
        new_data = new_img.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if data[x, y] in colors:
                    # Change the background color
                    new_data[x, y] = colors[data[x, y]]
                elif shape_color_rgb:
                    # Change the shape color while preserving the lightness
                    pixel_rgb = [v / 255 for v in data[x, y]]  # convert to range [0, 1]
                    pixel_hls = colorsys.rgb_to_hls(*pixel_rgb)
                    new_hls = (
                        shape_color_hls[0],
                        pixel_hls[1],
                        shape_color_hls[2],
                    )
                    new_rgb = [int(v * 255) for v in colorsys.hls_to_rgb(*new_hls)]
                    new_data[x, y] = tuple(new_rgb)
                else:
                    new_data[x, y] = data[x, y]

        new_img = new_img.resize(self.canvas_size)

        return apply_antialiasing(new_img) if self.antialiasing else new_img


def generate_all(
    output_folder=DEFAULTS[0]["output_folder"],
    canvas_size=DEFAULTS[0]["canvas_size"],
    background_color=DEFAULTS[0]["background_color"],
    antialiasing=DEFAULTS[0]["antialiasing"],
    stroke_color=DEFAULTS[0]["stroke_color"],
    behaviour_if_present=DEFAULTS[0]["behaviour_if_present"],
    object_longest_side=DEFAULTS[0]["object_longest_side"],
    shape_folder=DEFAULTS[0]["shape_folder"],
) -> str:
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    all_types = ["reference", "MP", "NAP"]
    output_folder = pathlib.Path(output_folder)
    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)
    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [(output_folder / i).mkdir(exist_ok=True, parents=True) for i in all_types]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "SampleName"])
        ds = DrawShape(
            background=background_color,
            canvas_size=canvas_size,
            antialiasing=antialiasing,
            obj_longest_side=object_longest_side,
        )

        for t in tqdm(all_types):
            for i in tqdm((pathlib.Path(shape_folder) / t).glob("*"), leave=False):
                name_sample = i.stem
                img_path = pathlib.Path(t) / f"{name_sample}.png"
                img = ds.process_image(
                    shape_folder / img_path,
                    stroke_color,
                )
                img.save(output_folder / img_path)
                writer.writerow([img_path, t, ds.background, name_sample])
    return str(output_folder)


if __name__ == "__main__":
    description = "3D Geon stimuli originally used in Kayaert et al. (2003) and obtained from https://geon.usc.edu/~ori. A feature dimension (such as the curvature of a Geon) is altered from a singular value (e.g. straight contour with 0 curvature) to two different values (e.g. slightly curved or very curved). The `reference` condition is the item with the intermediate value; the `MP change` condition consists of the sample with a greater non-singular value (that is, from slight curvature to greater curvature, which corresponds to a MP change), and the `NAP` change condition includes the samples with the singular value (that is, from slight curvature to non curvature, which corresponds to a NAP change).\nREF:Kayaert, Greet, Irving Biederman, and Rufin Vogels. 'Shape Tuning in Macaque Inferior Temporal Cortex'. Journal of Neuroscience 23, no. 7 (1 April 2003): 3016-27. https://doi.org/10.1523/JNEUROSCI.23-07-03016.2003."
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS[0]["output_folder"])

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS[0]["object_longest_side"],
        type=int,
        help="Specify the value in pixels to which the longest side of the line drawings will be resized (keeping the aspect ratio), before pasting the image into a canvas",
    )

    parser.add_argument(
        "--stroke_color",
        "-sc",
        default=DEFAULTS[0]["stroke_color"],
        help="Specify the color of the shape. The shading will be preserved. Leave it empty to not change the color of the shape. Specify it as a rgb tuple in the format of 255_255_255",
        type=lambda x: (
            [int(i) for i in x.split("_")]
            if "_" in x
            else x if isinstance(x, str) else x
        ),
    )
    parser.add_argument(
        "--shape_folder",
        "-sfolder",
        default=DEFAULTS[0]["shape_folder"],
        help="The folder containing the shapes.",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
