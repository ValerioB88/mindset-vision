import argparse
import uuid
import csv
import toml
import inspect

from tqdm import tqdm
from pathlib import Path
from src.tmp import crop
from src.utils.drawing_utils import DrawStimuli, resize_image_keep_aspect_ratio
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    check_download_ETH_80_dataset,
    delete_and_recreate_path,
)

import os
import sty
import pathlib
import glob
from PIL import Image
import numpy as np

category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
import re
from src.utils.misc import DEFAULTS as BASE_DEFAULTS
import uuid
import cv2

DEFAULTS = BASE_DEFAULTS.copy()

DEFAULTS.update(
    {
        "ETH_80_folder": "assets/ETH_80",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "object_longest_side": 200,
        "azimuth_lim": [0, 365],
        "inclination_lim": [30, 90],
    }
)


class DrawETH(DrawStimuli):
    def __init__(self, obj_longest_side, map_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side
        self.map_path = map_path

    def create_ETH(self, img_path):
        path_parts = img_path.split(os.sep)
        desired_path = os.path.join(*path_parts[-3:]).rstrip(".png")
        map_path = f"{self.map_path}/{desired_path}-map.png"

        map_pil = Image.open(map_path).convert("L")

        mask_array = np.array(map_pil)

        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        cropped_image = Image.open(img_path).crop((xmin, ymin, xmax, ymax))
        cropped_map_pil = map_pil.crop((xmin, ymin, xmax, ymax))

        canvas_only_obj = self.create_canvas(size=cropped_image.size)
        canvas_only_obj.paste(cropped_image, mask=cropped_map_pil)
        canvas_only_obj = Image.fromarray(
            resize_image_keep_aspect_ratio(
                np.array(canvas_only_obj), self.obj_longest_side
            )
        )

        canvas = self.create_canvas()
        canvas_only_obj
        paste_position = (
            (canvas.size[0] - canvas_only_obj.size[0]) // 2,
            (canvas.size[1] - canvas_only_obj.size[1]) // 2,
        )
        canvas.paste(canvas_only_obj, paste_position)

        return apply_antialiasing(canvas) if self.antialiasing else canvas


def generate_all(
    ETH_80_folder=DEFAULTS["ETH_80_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    object_longest_side=DEFAULTS["object_longest_side"],
    background_color=DEFAULTS["background_color"],
    inclination_lim=DEFAULTS["inclination_lim"],
    azimuth_lim=DEFAULTS["azimuth_lim"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    check_download_ETH_80_dataset(destination_dir=ETH_80_folder)
    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    ds = DrawETH(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        map_path=ETH_80_folder + "/maps/",
    )
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Class", "ObjectID", "Azimuth", "Inclination", "BackgroundColor"]
        )
        all_images = glob.glob(ETH_80_folder + "/images/*/*/*.png", recursive=True)
        for img_path in tqdm(all_images):
            class_num = Path(img_path).parts[2]
            object_id = int(Path(img_path).parts[3])
            match = re.search(
                r"([a-zA-Z]+)\d+-0*(\d+)-0*(\d+).png$", os.path.basename(img_path)
            )

            class_name = match.group(1)
            inclination = int(match.group(2))
            azimuth = int(match.group(3))

            # exclude extreme views (in practice, it's only a couple of views per object)
            # and it works more or less the same without excluding them.
            if not (inclination_lim[0] <= inclination <= inclination_lim[1]) or not (
                azimuth_lim[0] <= azimuth <= azimuth_lim[1]
            ):
                continue

            (output_folder / class_name / str(object_id)).mkdir(
                parents=True, exist_ok=True
            )
            img = ds.create_ETH(img_path)
            unique_hex = uuid.uuid4().hex[:8]

            img_save_path = (
                pathlib.Path(class_name)
                / str(object_id)
                / f"{Path(img_path).stem}_{unique_hex}.png"
            )
            img.save(output_folder / img_save_path)
            writer.writerow(
                [
                    img_save_path,
                    class_name,
                    object_id,
                    azimuth,
                    inclination,
                    ds.background,
                ]
            )
    return str(output_folder)


if __name__ == "__main__":
    description = "We use the ETH-80 dataset (https://github.com/chenchkx/ETH-80/tree/master), which contains 8 different categories (apples, cars, cows,cups, dogs, horses, pears and tomatoes), each consisting of 10 object instances, each object captured from 41 different viewpoints. The ETH-80 dataset is automatically downloaded when this dataset is generated, if not already present. "
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value to which the longest side of the object will be resized (keeping the aspect ratio),  before pasting the image into a canvas",
    )
    parser.add_argument(
        "--ETH_80_folder",
        "-ethf",
        help="A folder containing the original ETH-80 dataset. If the dataset is not present, it will be downloaded in this folder.",
        default=DEFAULTS["ETH_80_folder"],
    )

    parser.add_argument(
        "--inclination_lim",
        "-incl",
        help="Limits of the inclination viewpoints. For the ETH-80 dataset, they go from 0 (top-view) to 90 (plane-view). Specify whether you want to only consider some values between A and B inclusive. If provided as a command line argument, use a string in the format A_B, e.g. 45_90.",
        default=DEFAULTS["inclination_lim"],
        type=lambda x: [int(i) for i in x.split("_")],
    )
    parser.add_argument(
        "--azimuth_lim",
        "-azil",
        help="Limits of the azimuth viewpoints. For the ETH-80 dataset, they go from 0 to 338. Specify whether you want to only consider some values between A and B inclusive. If provided as a command line argument, use a string in the format A_B, e.g. 45_200.",
        default=DEFAULTS["azimuth_lim"],
        type=lambda x: [int(i) for i in x.split("_")],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
