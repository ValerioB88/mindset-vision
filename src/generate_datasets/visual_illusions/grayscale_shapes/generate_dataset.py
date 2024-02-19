import argparse
import uuid
import csv
import toml
import inspect

from tqdm import tqdm
from pathlib import Path
from .utils import ShapeConfigs, ColorPickerStimuli, add_arrow
from src.utils.misc import (
    apply_antialiasing,
    delete_and_recreate_path,
)
import os
import sty
import pathlib


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS = {
    "antialiasing": False,
    "behaviour_if_present": True,
    "num_samples": 5000,
    "canvas_size": [224, 224],
    "output_folder": f"data/{category_folder}/{name_dataset}",
}


def generate_all(
    num_samples=DEFAULTS["num_samples"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
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

    (output_folder / "all_images").mkdir(exist_ok=True, parents=True)
    shape_configs = ShapeConfigs()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Target Pixel Color", "Target Pixel Location"])

        for i in tqdm(range(num_samples)):
            img = ColorPickerStimuli(canvas_size)

            for step in range(20):
                shape_configs._refresh()
                random_shape_config = shape_configs._return_shape_config()

                getattr(img, "add_" + random_shape_config["shape"])(
                    **random_shape_config["parameters"]
                )

            propose_coordinates = img._propose_arrow_coord()
            counter = 0
            while (
                img._count_colors_withing_circle(propose_coordinates) > 1
                and counter < 100
            ):
                propose_coordinates = img._propose_arrow_coord()
                counter += 1

            if counter == 100:
                continue

            pixel_color = img._get_pixel_color(propose_coordinates)
            coord = tuple(
                map(lambda x: int(x * img.canvas.size[0]), propose_coordinates)
            )
            img.canvas = add_arrow(img.canvas, coord)
            img.canvas = apply_antialiasing(img) if antialiasing else img.canvas
            image_name = str(uuid.uuid4().hex[:8]) + ".png"
            img._shrink_and_save(save_as=output_folder / "all_images" / image_name)
            writer.writerow(
                [
                    str(Path("all_images") / image_name),
                    pixel_color,
                    coord,
                ]
            )


if __name__ == "__main__":
    description = "We include in MindSet: Vision two famous illusions related to lightness: the Lightness Contrast Illusion and the Adelson Checker Shadow Illusion. To facilitate testing for these and other lightness-related effects, we suggest using this dataset, whose purpose is not to elicit any illusion in humans or in DNNs but to train a network (or, with our suggested method, a decoder attached to a network) to output the grayscale value of a target pixel. Each image is composed of 20 overlapping items amongst the following types of shapes (circle, circle sector, circle segment, ellipse, rectangle with straight and rounded corners, heptagon, irregular polygon composed of a random number of edges from 3 to 10). Each shape is randomize according to their position, dimension, orientation, and grayscale filled color. We place 20 items to be sure that most space in the canvas is filled by some item, but in the end only few of them are visible. This results in a chaotic canvas with many different shapes with varying grayscale but with coherent pattern (as opposed to, for example, having each pixel of a different random grayscale value). In order to target a specific pixel to be predicted by the decoder, a small white vertical arrow (the `marker') of fixed size is placed randomly on the canvas. The arrow points to the pixel whose value can be used for prediction. Notice that while the images are commonly normalized from -1 to 1 before being fed into the network, the targeted pixel value to predict is in the 0-255 range."
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
        help="The size of the canvas. If called through command line, a string in the format NxM.",
        type=lambda x: (
            tuple([int(i) for i in x.split("x")]) if isinstance(x, str) else x
        ),
    )

    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=DEFAULTS["num_samples"],
        help="Each `sample` corresponds to an entire set of pair of shape_based_image_generation, for each condition.",
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

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
