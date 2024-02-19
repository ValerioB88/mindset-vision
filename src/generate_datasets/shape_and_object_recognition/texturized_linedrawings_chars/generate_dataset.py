import argparse
import csv
import os
import random

import cv2
import numpy as np
from pathlib import Path

import sty
from PIL import Image, ImageDraw
import math
import toml
import inspect

from torchvision.transforms import InterpolationMode, transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    delete_and_recreate_path,
    apply_antialiasing,
)
import uuid

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()

import random
import string

characters = string.ascii_letters + string.digits + string.punctuation


class DrawPatternedCanvas(DrawStimuli):
    def __init__(self, obj_longest_side, transform_code, *args, **kwargs):
        self.transform_code = transform_code
        super().__init__(*args, **kwargs)
        self.obj_longest_side = obj_longest_side

    def get_canvas_char_pattered(
        self,
        size,
        tile_char,
        font_size,
        spacing=0,
        rotation_angle=45,
        font_path="assets/arial.ttf",
        background=None,
    ):
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple(
                [np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2
            ),
            background=background,
        )

        bbox = font.getbbox(tile_char + " " * spacing)
        char_width = bbox[2]
        char_heights = bbox[3]
        sp = max(char_width, char_heights)

        width, height = img.size
        num_x = width // char_width + 2
        draw = ImageDraw.Draw(img)
        tile_string = (tile_char + " " * spacing) * num_x

        for y in range(-20, height, char_heights):
            draw.text(
                (1, y),
                tile_string,
                fill=self.line_args["fill"],
                font=font,
            )
        if rotation_angle > 0:
            img = img.rotate(rotation_angle, resample=Image.Resampling.NEAREST)

        img = transforms.CenterCrop((size[1], size[0]))(img)
        return img

    def draw_pattern(
        self,
        img_path,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        expand_factor = 1
        opencv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        opencv_img = resize_image_keep_aspect_ratio(opencv_img, self.obj_longest_side)

        mask = get_mask_from_linedrawing(opencv_img, fill=True)
        canvas = self.get_canvas_char_pattered(
            size=tuple(np.array(self.canvas_size) * expand_factor),
            tile_char=background_char,
            font_size=background_font_size,
            rotation_angle=np.rad2deg(rotation_angle_rad),
            spacing=0,
        )

        perpendicular_radian = (
            (rotation_angle_rad + math.pi / 2)
            if abs(rotation_angle_rad + math.pi / 2) <= math.pi / 2
            else (rotation_angle_rad - math.pi / 2)
        )
        canvas_foreg_text = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
            background=self.background,
        )

        canvas.paste(
            canvas_foreg_text,
            (
                canvas.size[0] // 2 - mask.size[0] // 2,
                canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas) if self.antialiasing else canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS.update(
    {
        "linedrawing_input_folder": "assets/baker_2018_linedrawings/cropped/",
        "num_samples": 500,
        "object_longest_side": 200,
        "background_char": " ",
        "foreground_char": "random",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
        "font_size": [15, 20],
    }
)


def generate_all(
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    num_samples=DEFAULTS["num_samples"],
    object_longest_side=DEFAULTS["object_longest_side"],
    background_char=DEFAULTS["background_char"],
    foreground_char=DEFAULTS["foreground_char"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    font_size=DEFAULTS["font_size"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    # transf_code = {"translation": [-0.1, 0.1]}
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

    ds = DrawPatternedCanvas(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        width=1,
        transform_code=None,  # transf_code,
    )
    jpg_files = list(linedrawing_input_folder.rglob("*.jpg"))
    png_files = list(linedrawing_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Class",
                "BackgroundColor",
                "BackgroundChar",
                "ForegroundChar",
                "FontSize",
                "RotationAngle",
                "IterNum",
                # "BackgroundSpacing",
            ]
        )
        for img_path in tqdm(image_files):
            print(img_path)
            for n in tqdm(range(num_samples), leave=False):
                class_name = img_path.parent.stem
                rotation_angle = random.randint(-60, 60)
                font_s = (
                    random.randint(font_size[0], font_size[1])
                    if isinstance(font_size, list)
                    else font_size
                )

                background_c = (
                    random.choice(characters)
                    if background_char == "random"
                    else background_char
                )
                foreground_c = (
                    random.choice(characters)
                    if foreground_char == "random"
                    else foreground_char
                )

                img = ds.draw_pattern(
                    img_path=img_path,
                    background_char=background_c,
                    foreground_char=foreground_c,
                    background_font_size=font_s,
                    foreground_font_size=font_s,
                    rotation_angle_rad=np.deg2rad(rotation_angle),
                )
                unique_id = uuid.uuid4().hex[:8]
                image_name = img_path.stem
                path = Path(class_name) / f"{image_name}_{unique_id}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        class_name,
                        ds.background,
                        background_c,
                        foreground_c,
                        font_s,
                        rotation_angle,
                        n,
                    ]
                )
    return str(output_folder)


if __name__ == "__main__":
    description = "The dataset consists of texturized familiar objects by using as a base items the line drawings from Baker et al. (2018), but the user can specify a different folder oe line drawings (which should be images of black strokes on a white background). The texturization consists of masking the internal contour of a line drawing/silhouette with a pattern of a repeated character with a randomized font size, rotated by a random degree. The character is randomly selected between an letters, digits, or punctuation. The user can specify the texturization of the background as well, although we have found that doing so will turn object recognition from trivial to very  challenging, depending on the selected character, and thus suggest not using it.\nREF: Baker, Nicholas, Hongjing Lu, Gennady Erlikhman, and Philip J. Kellman. 'Deep Convolutional Networks Do Not Classify Based on Global Object Shape'. PLoS Computational Biology 14, no. 12 (2018): 1-43. https://doi.org/10.1371/journal.pcbi.1006613."
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--num_samples",
        "-ns",
        default=DEFAULTS["num_samples"],
        help="The number of augmented samples to generate for each line drawings",
        type=int,
    )

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio),  before pasting the image into a canvas",
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )

    parser.add_argument(
        "--background_char",
        "-bgch",
        default=DEFAULTS["background_char"],
        help="The character to be used as background. Use `random` to use a random character for each sample",
    )

    parser.add_argument(
        "--foreground_char",
        "-fgch",
        default=DEFAULTS["foreground_char"],
        help="The character to be used as foreground. Write `random` to use a different character for each image",
    )

    parser.add_argument(
        "--font_size",
        "-fs",
        help="If a number, it defines the size of the font for all images. It can be a string in the form A_B, in which case the size will be drawn from a uniform(A, B) distribution for each image",
        default=DEFAULTS["font_size"],
        type=lambda x: (list([int(i) for i in x.split("_")]) if "_" in x else list(x)),
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


###
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch \ -fgch .
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch . -fgch ""
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch "" -fgch .
