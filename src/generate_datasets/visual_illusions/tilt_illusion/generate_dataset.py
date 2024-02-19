import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import sty
from PIL import Image
from PIL import ImageDraw
import toml
import inspect

from tqdm import tqdm
import uuid
from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def generate_grating(canvas_size, frequency, orientation, phase=0):
    width, height = canvas_size
    # Generate x and y coordinates
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    x, y = np.meshgrid(x, y)

    # Rotate the grid by the specified orientation
    x_prime = x * np.cos(orientation) - y * np.sin(orientation)

    # Create the sinusoidal grating
    grating = 0.5 * (1 + np.sin(frequency * x_prime + phase))
    return grating


all_pil_images = []
freq = 10


class DrawTiltIllusion(DrawStimuli):
    def generate_illusion(
        self, theta_center, radius, center_test, freq, theta_context=None
    ):
        if theta_context is not None:
            context = generate_grating(self.canvas_size, freq, theta_context)
            context = Image.fromarray(np.uint8(context * 255))
        else:
            context = self.create_canvas()
        if theta_center is not None:
            center = generate_grating(self.canvas_size, freq, theta_center)
            center = Image.fromarray(np.uint8(center * 255))
        else:
            center = self.create_canvas()
        mask = Image.new("L", center.size, 0)

        draw = ImageDraw.Draw(mask)
        center_test = np.array(center_test) * self.canvas_size
        draw.ellipse(
            (
                center_test[0] - radius,
                center_test[1] - radius,
                center_test[0] + radius,
                center_test[1] + radius,
            ),
            fill=255,
        )

        context.paste(center, mask=mask)
        return apply_antialiasing(context) if self.antialiasing else context


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "num_samples_only_center": 1000,
        "num_samples_only_context": 1000,
        "num_samples_center_context": 1000,
        "output_folder": f"data/{category_folder}/{name_dataset}",
    }
)


def generate_all(
    num_samples_only_center=DEFAULTS["num_samples_only_center"],
    num_samples_only_context=DEFAULTS["num_samples_only_context"],
    num_samples_center_context=DEFAULTS["num_samples_center_context"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    def get_random_values():
        size_scale = np.random.uniform(0.1, 0.6)
        radius = canvas_size[0] // 2 * size_scale
        center = (
            np.random.uniform(radius, canvas_size[0] - radius) // canvas_size[0],
            np.random.uniform(radius, canvas_size[1] - radius) // canvas_size[1],
        )
        freq = random.randint(5, 20)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        return theta, radius, center, freq

    output_folder = Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [
        (output_folder / i).mkdir(exist_ok=True, parents=True)
        for i in ["only_center", "only_context", "center_context"]
    ]

    ds = DrawTiltIllusion(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "BackgroundColor",
                "ThetaCenter",
                "Radius",
                "Frequency",
                "ThetaContext",
                "IterNum",
            ]
        )
        for i in tqdm(range(num_samples_only_center)):
            unique_hex = uuid.uuid4().hex[:8]

            theta_center, radius, _, freq = get_random_values()
            path = Path("only_center") / f"{-theta_center:.3f}_0_{unique_hex}.png"
            img = ds.generate_illusion(theta_center, radius, (0.5, 0.5), freq)
            img.save(str(output_folder / path))
            writer.writerow(
                [path, "only_center", ds.background, theta_center, radius, freq, "", i]
            )
        for i in tqdm(range(num_samples_only_context)):
            unique_hex = uuid.uuid4().hex[:8]

            theta_context, radius, _, freq = get_random_values()
            path = Path("only_context") / f"{-theta_center:.3f}_0_{unique_hex}.png"
            img = ds.generate_illusion(None, radius, (0.5, 0.5), freq, theta_context)
            img.save(str(output_folder / path))
            writer.writerow(
                [path, "only_context", ds.background, 0, radius, freq, theta_context, i]
            )

        all_thetas = np.linspace(-np.pi / 2, np.pi / 2, num_samples_center_context)
        for i, theta_context in enumerate(tqdm(all_thetas)):
            _, radius, _, freq = get_random_values()
            img = ds.generate_illusion(0, radius, (0.5, 0.5), freq, theta_context)
            unique_hex = uuid.uuid4().hex[:8]
            path = Path("center_context") / f"0_{theta_context:.3f}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow(
                [
                    path,
                    "center_context",
                    ds.background,
                    0,
                    radius,
                    freq,
                    theta_context,
                    i,
                ]
            )
    return str(output_folder)


if __name__ == "__main__":
    description = "We provide one illusory condition, in which an oriented grating pattern is presented within a circular mask (`center grating') and a differently oriented grating is placed as the background (`context' grating); and two non-illusory conditions: one in which the background is uniformly colored and only a center mask contains the oriented grating pattern; and vice versa. The samples are varied in their orientation and frequency of the gratings, and in the size of the central grating."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_only_center",
        "-ncenter",
        help="Number of samples with only the center grating",
        default=DEFAULTS["num_samples_only_center"],
        type=int,
    )
    parser.add_argument(
        "--num_samples_only_context",
        "-ncontext",
        help="Number of samples with only the context grating",
        default=DEFAULTS["num_samples_only_context"],
        type=int,
    )
    parser.add_argument(
        "--num_samples_center_context",
        "-nboth",
        default=DEFAULTS["num_samples_center_context"],
        help="Number of samples for center and context grating",
        type=int,
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
