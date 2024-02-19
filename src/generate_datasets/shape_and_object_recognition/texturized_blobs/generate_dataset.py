import argparse
import csv
import os
import random

import cv2
import toml
import inspect

from src.utils.shape_based_image_generation.modules.shapes import Shapes
import numpy as np
from pathlib import Path

import sty
from PIL import Image, ImageDraw
import math
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from src.utils.shape_based_image_generation.modules.parent import ParentStimuli

from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
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
    ):
        font = ImageFont.truetype(font_path, font_size)
        img = self.create_canvas(
            size=tuple([np.round(np.sqrt(size[0] ** 2 + size[1] ** 2)).astype(int)] * 2)
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

    def draw_blob(self, blob_id):
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color(self.fill)

        blob.add_puddle(size=0.2, seed=blob_id)

        blob.register()
        self.create_canvas()  # dummy call to update the background for rnd-uniform mode
        parent.add_background(self.background)

        return apply_antialiasing(parent.canvas) if self.antialiasing else parent.canvas

    def draw_pattern(
        self,
        blob_id,
        background_char,
        background_font_size,
        rotation_angle_rad,
        foreground_char,
        foreground_font_size,
    ):
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=1,
        )

        blob = Shapes(parent)
        blob.set_color((0, 0, 0, 255))
        blob.add_puddle(size=0.2, seed=blob_id)
        blob.register()
        self.create_canvas()  # dummy call to update the background for rnd-uniform mode
        parent.add_background((255, 255, 255))

        mask = get_mask_from_linedrawing(
            np.array(parent.canvas.convert("L")), fill=True
        )
        expand_factor = 1
        canvas_bg = self.get_canvas_char_pattered(
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
        canvas_fg = self.get_canvas_char_pattered(
            size=mask.size,
            tile_char=foreground_char,
            font_size=foreground_font_size,
            rotation_angle=np.rad2deg(perpendicular_radian),
            spacing=0,
        )

        canvas_bg.paste(
            canvas_fg,
            (
                canvas_bg.size[0] // 2 - mask.size[0] // 2,
                canvas_bg.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )
        return apply_antialiasing(canvas_bg) if self.antialiasing else canvas_bg


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS.update(
    {
        "num_samples_per_blob": 5,
        "num_blobs": 10,
        "object_longest_side": 200,
        "background_char": " ",
        "foreground_char": "random",
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "antialiasing": False,
        "font_size": [15, 20],
    }
)


def generate_all(
    num_samples_per_blob=DEFAULTS["num_samples_per_blob"],
    num_blobs=DEFAULTS["num_blobs"],
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
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    ds = DrawPatternedCanvas(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        obj_longest_side=object_longest_side,
        width=1,
        transform_code=None,  # transf_code,
    )
    (output_folder / "blobs").mkdir(exist_ok=True, parents=True)
    (output_folder / "texturized_blobs").mkdir(exist_ok=True, parents=True)

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "BlobID",
                "IsTexturized",
                "BackgroundColor",
                "BackgroundChar",
                "ForegroundChar",
                "FontSize",
                "RotationAngle",
                "IterNum",
                # "BackgroundSpacing",
            ]
        )
        for blob_id in tqdm(range(num_blobs)):
            img = ds.draw_blob(blob_id)
            path = Path("blobs") / f"{blob_id}.png"

            img.save(str(output_folder / path))

            writer.writerow(
                [
                    path,
                    blob_id,
                    False,
                    ds.background,
                    "",
                    "",
                    "",
                    "",
                    0,
                ]
            )
            (Path(output_folder) / "texturized_blobs" / str(blob_id)).mkdir(
                exist_ok=True, parents=True
            )
            for n in tqdm(range(num_samples_per_blob), leave=False):
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
                    blob_id=blob_id,
                    background_char=background_c,
                    foreground_char=foreground_c,
                    background_font_size=font_s,
                    foreground_font_size=font_s,
                    rotation_angle_rad=np.deg2rad(rotation_angle),
                )
                hex_id = uuid.uuid4().hex[:8]

                path = Path("texturized_blobs") / str(blob_id) / f"{hex_id}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [
                        path,
                        blob_id,
                        True,
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
    description = "The dataset consists of texturized familiar objects, which consists of randomly generated blob-like shapes. The texturization consists of masking the internal contour of a line drawing/silhouette with a pattern of a repeated character with a randomized font size, rotated by a random degree. The character is randomly selected between an letters, digits, or punctuation. The user can specify the texturization of the background as well, although we have found that doing so will turn object recognition from trivial to very  challenging, depending on the selected character, and thus suggest not using it. The user can specify the number of blobs to generate and texturize."
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.add_argument(
        "--num_samples_per_blob",
        "-ns",
        default=DEFAULTS["num_samples_per_blob"],
        help="The number of augmented samples to generate for each blob",
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
        type=lambda x: ([[int(i) for i in x.split("_")]] if "_" in x else x),
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)


###
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch \ -fgch .
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch . -fgch ""
# -ntest 10 -csize 224x224 -objsize 150x150 -bgch "" -fgch .
