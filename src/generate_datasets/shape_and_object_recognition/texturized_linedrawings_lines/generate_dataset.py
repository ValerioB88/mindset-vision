import toml
import inspect

from tqdm import tqdm
import argparse
import csv
import os
import random
import numpy as np
from pathlib import Path
import cv2
import sty
import math
from torchvision.transforms import InterpolationMode, transforms
from PIL import ImageDraw

from src.utils.misc import my_affine
from src.utils.drawing_utils import (
    DrawStimuli,
    get_mask_from_linedrawing,
    resize_image_keep_aspect_ratio,
)
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
    get_affine_rnd_fun,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


class DrawPatternedCanvas(DrawStimuli):
    def add_line_pattern(self, canvas, line_length=10, slope_rad=0, density=1):
        width, height = canvas.size

        draw = ImageDraw.Draw(canvas)

        line_segment = [
            (0, 0),
            (line_length * math.cos(slope_rad), line_length * math.sin(slope_rad)),
        ]
        horizontal_spacing = int(np.round(line_length / density))
        vertical_spacing = int(np.round(line_length / density))
        noise = 1
        for y in range(
            0,
            height,
            np.round(
                line_length * np.abs(math.sin(slope_rad)) + vertical_spacing
            ).astype(int),
        ):
            for x in range(
                0,
                width,
                np.round(line_length * math.cos(slope_rad) + horizontal_spacing).astype(
                    int
                ),
            ):
                x_offset, y_offset = random.randint(-noise, noise), random.randint(
                    -noise, noise
                )
                shifted_line_segment = [
                    (point[0] + x + x_offset, point[1] + y + y_offset)
                    for point in line_segment
                ]
                draw.line(shifted_line_segment, **self.line_args)
        return canvas

    def __init__(
        self,
        texturize_background,
        texturize_foreground,
        obj_longest_side,
        transform_code,
        density,
        *args,
        **kwargs,
    ):
        self.obj_longest_side = obj_longest_side
        self.density = density
        self.transform_code = transform_code
        self.texturize_foreground = texturize_foreground
        self.texturize_background = texturize_background
        super().__init__(*args, **kwargs)

    def draw_pattern(
        self,
        img_path,
        slope_rad,
        line_length,
    ):
        expand_factor = 1.5
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = get_mask_from_linedrawing(
            resize_image_keep_aspect_ratio(img, self.obj_longest_side), fill=True
        )

        bg_canvas = self.create_canvas(
            size=tuple((np.array(self.canvas_size) * expand_factor).astype(int))
        )
        if self.texturize_background:
            bg_canvas = self.add_line_pattern(
                canvas=bg_canvas,
                line_length=line_length,
                slope_rad=slope_rad,
                density=self.density,
            )
        perpendicular_radian = (
            (slope_rad + math.pi / 2)
            if abs(slope_rad + math.pi / 2) <= math.pi / 2
            else (slope_rad - math.pi / 2)
        )

        fg_canvas = self.create_canvas(size=mask.size, background=self.background)
        if self.texturize_foreground:
            canvas_foreg_text = self.add_line_pattern(
                fg_canvas,
                line_length,
                perpendicular_radian,
                self.density,
            )

        bg_canvas.paste(
            canvas_foreg_text,
            (
                bg_canvas.size[0] // 2 - mask.size[0] // 2,
                bg_canvas.size[1] // 2 - mask.size[1] // 2,
            ),
            mask=mask,
        )

        af = get_affine_rnd_fun(self.transform_code)()
        img = my_affine(
            bg_canvas,
            translate=list(np.array(af["tr"]) / expand_factor),
            angle=af["rt"],
            scale=af["sc"],
            shear=af["sh"],
            interpolation=InterpolationMode.NEAREST,
            fill=self.background,
        )
        img = transforms.CenterCrop((self.canvas_size[1], self.canvas_size[0]))(img)
        return apply_antialiasing(img) if self.antialiasing else img


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS.update(
    {
        "linedrawing_input_folder": "assets/baker_2018_linedrawings/cropped/",
        "num_samples": 500,
        "object_longest_side": 200,
        "density": 1.8,
        "antialiasing": False,
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "texturize_foreground": True,
        "texturize_background": False,
    }
)


def generate_all(
    linedrawing_input_folder=DEFAULTS["linedrawing_input_folder"],
    num_samples=DEFAULTS["num_samples"],
    object_longest_side=DEFAULTS["object_longest_side"],
    density=DEFAULTS["density"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
    texturize_background=DEFAULTS["texturize_background"],
    texturize_foreground=DEFAULTS["texturize_foreground"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    transf_code = {"translation": [-0.1, 0.1]}
    linedrawing_input_folder = Path(linedrawing_input_folder)
    output_folder = Path(output_folder)

    all_categories = [
        os.path.splitext(os.path.basename(i))[0]
        for i in linedrawing_input_folder.glob("*")
    ]

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [(output_folder / cat).mkdir(exist_ok=True, parents=True) for cat in all_categories]
    ds = DrawPatternedCanvas(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        texturize_background=texturize_background,
        texturize_foreground=texturize_foreground,
        obj_longest_side=object_longest_side,
        density=density,
        width=1,
        transform_code=transf_code,  # np.max((1, np.round(canvas_size[0] * 0.00446).astype(int))),
    )
    jpg_files = list(linedrawing_input_folder.rglob("*.jpg"))
    png_files = list(linedrawing_input_folder.rglob("*.png"))

    image_files = jpg_files + png_files
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            ["Path", "Class", "BackgroundColor", "SlopeLine", "LineLength", "IterNum"]
        )
        for img_path in tqdm(image_files):
            print(img_path)
            class_name = img_path.parent.stem
            image_name = img_path.stem
            for n in tqdm(range(num_samples), leave=False):
                slope_line = np.deg2rad(random.uniform(*random.choice([(-60, 60)])))
                line_length = random.randint(4, 8)
                img = ds.draw_pattern(
                    img_path,
                    slope_line,
                    line_length,
                )
                # We don't perform scaling or rotation because it creates many artifacts that make the task too difficult even for humans.
                path = Path(class_name) / f"{image_name}_{n}.png"
                img.save(output_folder / path)
                writer.writerow(
                    [path, class_name, ds.background, slope_line, line_length, n]
                )
    return str(output_folder)


if __name__ == "__main__":
    description = "The dataset consists of texturized familiar objects by using as a base items the line drawings from Baker et al. (2018), but the user can specify a different folder oe line drawings (which should be images of black strokes on a white background). The texturization consists of lines placed at random degree. This is an alternative texturization method to the 'texturized_linedrawings_chars'. We suggest using the 'chars' method instead.  \nREF: Baker, Nicholas, Hongjing Lu, Gennady Erlikhman, and Philip J. Kellman. 'Deep Convolutional Networks Do Not Classify Based on Global Object Shape'. PLoS Computational Biology 14, no. 12 (2018): 1-43. https://doi.org/10.1371/journal.pcbi.1006613."
    parser = argparse.ArgumentParser(description=description)
    add_general_args(parser)
    parser.set_defaults(antialiasing=DEFAULTS["antialiasing"])
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=DEFAULTS["num_samples"],
        help="Number of different texturized sample for each image",
    )
    parser.add_argument(
        "--density",
        "-d",
        default=DEFAULTS["density"],
        help="The desity of the pattern. The horizontal and vertical spacing are equal to line_length/density",
    )

    parser.add_argument(
        "--object_longest_side",
        "-objlside",
        default=DEFAULTS["object_longest_side"],
        type=int,
        help="Specify the value to which the longest side of the line drawings will be resized (keeping the aspect ratio),  before pasting the image into a canvas",
    )
    parser.add_argument(
        "--texturize_background",
        "-txtbg",
        default=DEFAULTS["texturize_background"],
        type=int,
        help="Whether texturize the background",
    )
    parser.add_argument(
        "--texturize_foreground",
        "-txtfg",
        default=DEFAULTS["texturize_foreground"],
        type=int,
        help="Whether texturize the foreground",
    )
    parser.add_argument(
        "--linedrawing_input_folder",
        "-fld",
        dest="linedrawing_input_folder",
        help="A folder containing linedrawings. We assume these to be black strokes-on-white canvas simple contour drawings.",
        default=DEFAULTS["linedrawing_input_folder"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
