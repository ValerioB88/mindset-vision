import argparse
import csv
import math
import os
import random
from pathlib import Path
import uuid

import numpy as np
import sty
from PIL import ImageDraw
import toml
import inspect

from tqdm import tqdm

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def draw_arrow(draw, pos, theta, angle_arrow, arrow_length, width, color):
    x, y = pos
    arrow_theta1 = theta - angle_arrow
    arrow_theta2 = theta + angle_arrow

    arrow_end_x1 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta1)))
    arrow_end_y1 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta1)))
    arrow_end_x2 = x + np.round(arrow_length * math.cos(math.radians(arrow_theta2)))
    arrow_end_y2 = y + np.round(arrow_length * math.sin(math.radians(arrow_theta2)))

    # Draw the arrow lines
    draw.line([(x, y), (arrow_end_x1, arrow_end_y1)], fill=color, width=width)
    draw.line([(x, y), (arrow_end_x2, arrow_end_y2)], fill=color, width=width)


class DrawMullerLyer(DrawStimuli):
    def generate_illusion(
        self,
        line_position_rel,
        line_length,
        arrow_angle,
        arrow_cap_angle,
        arrow_length,
        type,
    ):
        get_arrow_rnd_pos = lambda: (
            random.randint(arrow_length, self.canvas_size[0] - arrow_length),
            random.randint(arrow_length, self.canvas_size[1] - arrow_length),
        )

        img = self.create_canvas()
        d = ImageDraw.Draw(img)
        line_position = tuple(
            (np.array(line_position_rel) * self.canvas_size).astype(int)
        )
        if type == "scrambled":
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                get_arrow_rnd_pos(),
                theta=arrow_angle + 180,
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            d.line(
                (
                    np.round(line_position[0] - line_length // 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length // 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )
        else:
            d.line(
                (
                    np.round(line_position[0] - line_length / 2).astype(int),
                    line_position[1],
                    np.round(line_position[0] + line_length / 2).astype(int),
                    line_position[1],
                ),
                **self.line_args,
            )

            draw_arrow(
                d,
                (line_position[0] - line_length // 2, line_position[1]),
                theta=(180 if type == "outward" else 0),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
            draw_arrow(
                d,
                (line_position[0] + line_length // 2, line_position[1]),
                theta=(0 if type == "outward" else 180),
                angle_arrow=arrow_cap_angle,
                arrow_length=arrow_length,
                color=self.fill,
                width=self.line_args["width"],
            )
        return apply_antialiasing(img) if self.antialiasing else img


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "num_samples_scrambled": 5000,
        "num_samples_illusory": 500,
        "output_folder": f"data/{category_folder}/{name_dataset}",
    }
)


def generate_all(
    num_samples_scrambled=DEFAULTS["num_samples_scrambled"],
    num_samples_illusory=DEFAULTS["num_samples_illusory"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
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

    conditions = ["scrambled", "inward", "outward"]
    [(output_folder / i).mkdir(exist_ok=True, parents=True) for i in conditions]

    ds = DrawMullerLyer(
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
        width=1,
    )

    def get_random_params():
        line_length = random.randint(
            int(canvas_size[0] * 0.25), int(canvas_size[0] * 0.67)
        )
        arrow_length = random.randint(
            int(canvas_size[0] * 0.07), int(canvas_size[0] * 0.134)
        )
        line_position = tuple(
            np.array(
                [
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[0] - arrow_length - line_length // 2,
                    ),
                    random.randint(
                        arrow_length + line_length // 2,
                        canvas_size[1] - arrow_length - line_length // 2,
                    ),
                ],
            )
            / canvas_size
        )
        cap_arrows_angle = random.randint(
            int(canvas_size[0] * 0.045), int(canvas_size[1] * 0.2)
        )
        angle_arrow = random.randint(0, 360)
        return line_length, line_position, arrow_length, cap_arrows_angle, angle_arrow

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Type",
                "BackgroundColor",
                "LineLength",
                "LinePosition",
                "ArrowLength",
                "CapArrowAngle",
                "ArrowAngle",
                "IterNum",
            ]
        )
        for c in tqdm(conditions):
            num_samples = (
                num_samples_scrambled if c == "scrambled" else num_samples_illusory
            )
            for i in tqdm(range(num_samples), leave=False):
                (
                    line_length,
                    line_position,
                    arrow_length,
                    cap_arrow_angle,
                    arrow_angle,
                ) = get_random_params()

                img = ds.generate_illusion(
                    line_position_rel=line_position,
                    line_length=line_length,
                    arrow_angle=arrow_angle,
                    arrow_cap_angle=cap_arrow_angle,
                    arrow_length=arrow_length,
                    type=c,
                )
                unique_hex = uuid.uuid4().hex[:8]
                path = Path(c) / f"{line_length}_{unique_hex}.png"
                img.save(str(output_folder / path))
                writer.writerow(
                    [
                        path,
                        c,
                        ds.background,
                        line_length,
                        line_position,
                        arrow_length,
                        cap_arrow_angle,
                        arrow_angle,
                        i,
                    ]
                )
    return str(output_folder)


if __name__ == "__main__":
    description = "The Müller-Lyer illusion stimuli are procedurally generated in one of two `illusory' configurations (with inward or outward `fins') or in a `scrambled' configuration. In the latter, the fins are arranged randomly in the canvas, separated from the line segment. In all three conditions, we vary the line length, the position of the line, and the angle of the fins."
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=description
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_scrambled",
        "-nscr",
        default=DEFAULTS["num_samples_scrambled"],
        help="Number of samples for the scrambled configuration, in which the arrow caps and the lines are randomly placed in the canvas",
        type=int,
    )
    parser.add_argument(
        "--num_samples_illusory",
        "-nill",
        default=DEFAULTS["num_samples_illusory"],
        help="Number of samples for the illusory configuration, with the standard Muller-Lyer illusion",
        type=int,
    )
    parser.set_defaults(antialiasing=False)
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
