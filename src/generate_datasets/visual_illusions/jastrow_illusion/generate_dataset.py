import argparse
import csv
import math
import os
import pathlib

import sty
import toml
import inspect

from tqdm import tqdm

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import add_general_args, delete_and_recreate_path
from src.utils.shape_based_image_generation.modules.parent import ParentStimuli
from src.utils.shape_based_image_generation.modules.shapes import Shapes
import uuid
import numpy as np
from pathlib import Path
import random

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


def get_random_params():
    arc_curvature = np.random.uniform(45, 60)
    width = np.random.uniform(3e-2, 2e-1)
    size_top = np.random.uniform(0.01, 0.06)
    size_bottom = np.random.uniform(0.01, 0.06)
    return arc_curvature, width, size_top, size_bottom


class JastrowParent(ParentStimuli):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_jastrow_factor(self):
        """
        Since we need a way to approximate how much an image will trigger the Jastrow illusion without having to
        actually show the image to a human, I propose the following method - we use a "Jastrow" factor, which is a
        number between 0 and 1, where larger numbers indicate a higher likelihood of triggering the illusion.

        The Jastrow factor is calculated as the product of the rotation similarity and the spatial similarity.
        """
        assert (
            len(self.contained_shapes) == 2
        ), "Make sure there are only two arc shapes"
        assert all(
            [shape.shape == "arc" for shape in self.contained_shapes]
        ), "Shapes must be arc"

        rotations = [shape.rotation for shape in self.contained_shapes]
        positions = [shape.position for shape in self.contained_shapes]

        # calculate the rotation similarity using the cosine similarity
        rotation_similarity = self.get_rotation_similarity(*rotations)

        # calculate the spatial similarity by calculating the distance between the two shapes and dividing by the max distance
        self.position_similarity = np.linalg.norm(
            np.array(positions[0]) - np.array(positions[1])
        ) / math.sqrt(2)
        self.position_similarity = 1 - self.position_similarity

        # calculate the Jastrow factor as the product of rotation similarity and spatial similarity
        jastrow_factor = rotation_similarity * self.position_similarity
        return jastrow_factor

    def get_rotation_similarity(self, rotation1, rotation2):
        cos_sim = math.cos(math.radians(rotation1 - rotation2))
        return (cos_sim + 1) / 2


class DrawJastrow(DrawStimuli):
    def generate_jastrow_illusion(
        self, arc, width, size_red, size_blue, top_color, type_stimulus
    ):
        correct_stimulus = False
        while not correct_stimulus:
            position_fun = lambda: (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            rotation_fun = lambda: random.uniform(0, 360)
            parent = JastrowParent(
                target_image_size=self.canvas_size,
                initial_expansion=4 if self.antialiasing else 1,
            )
            # With illusory or aligned, arc_1 is always the top 1. Then "on_top" decides it's color.
            arcs_sizes = (
                [size_red, size_blue] if top_color == "red" else [size_blue, size_red]
            )
            if type_stimulus == "random_same_size":
                arcs_sizes = [np.mean(arcs_sizes)] * 2
            arc_1 = Shapes(parent=parent)
            arc_1.add_arc(size=arcs_sizes[0], arc=arc, width=width)
            (
                arc_1.move_to(position_fun()).rotate(rotation_fun())
                if "random" in type_stimulus
                else None
            )

            arc_2 = Shapes(parent=parent)
            arc_2.add_arc(size=arcs_sizes[1], arc=arc, width=width)
            (
                arc_2.move_to(position_fun()).rotate(rotation_fun())
                if "random" in type_stimulus
                else None
            )
            parent.center_shapes()

            if type_stimulus == "aligned" or type_stimulus == "illusory":
                arc_1.move_next_to(arc_2, "UP")
                arc_1.set_color(top_color).register()
                arc_2.set_color({"red": "blue", "blue": "red"}[top_color]).register()
            elif "random" in type_stimulus:
                while (
                    arc_1.is_touching(arc_2)
                    or parent.compute_jastrow_factor().round(3) > 0.7
                ):
                    arc_1 = arc_1.move_to(position_fun()).rotate(rotation_fun())
                    arc_2 = arc_2.move_to(position_fun()).rotate(rotation_fun())
                arc_1.set_color("red").register()
                arc_2.set_color("blue").register()

            self.create_canvas()  # dummy call to update the background for rnd-uniform mode
            parent.add_background(self.background)
            parent.shrink() if self.antialiasing else None

            if parent.count_pixels("red") > 0 and parent.count_pixels("blue") > 0:
                correct_stimulus = True
        return parent.canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "num_samples_illusory": 50,
        "num_samples_random": 1000,
        "num_samples_aligned": 50,
        "num_samples_random_same_size": 50,
        "output_folder": f"data/{category_folder}/{name_dataset}",
    }
)


def generate_all(
    num_samples_illusory=DEFAULTS["num_samples_illusory"],
    num_samples_random=DEFAULTS["num_samples_random"],
    num_samples_aligned=DEFAULTS["num_samples_aligned"],
    num_samples_random_same_size=DEFAULTS["num_samples_random_same_size"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    output_folder = pathlib.Path(output_folder)

    # Aligned means the shapes are one on top of the other but different sizes
    # Illusory is Aligned with the same size
    # Random is not aligned, different sizes
    # random_same_size means not aligned, but two shapes on the same image are of the same size
    types = ["illusory", "random_same_size", "random", "aligned"]

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    on_top_cols = ["red", "blue"]

    [
        [
            (
                output_folder / type / ("" if "random" in type else f"{top}_on_top")
            ).mkdir(exist_ok=True, parents=True)
            for type in types
        ]
        for top in on_top_cols
    ]

    ds = DrawJastrow(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "BackgroundColor",
                "Type",
                "ArcSize",
                "ArcWidth",
                "SizeTop",
                "SizeBottom",
                "OnTop",
                "SampleNum",
                "SizeRed",
                "SizeBlue",
                "SizeRedMinusBlue",
            ]
        )

        for type in tqdm(types):
            num_samples = {
                "illusory": num_samples_illusory,
                "aligned": num_samples_aligned,
                "random": num_samples_random,
                "random_same_size": num_samples_random_same_size,
            }[type]
            for idx in tqdm(range(num_samples), leave=False):
                for top_color in on_top_cols:
                    arc_curvature, width, size_red, size_blue = get_random_params()
                    size_blue = size_red if type == "illusory" else size_blue
                    img = ds.generate_jastrow_illusion(
                        arc_curvature,
                        width,
                        size_red,
                        size_blue,
                        top_color,
                        type_stimulus=type,
                    )
                    unique_hex = uuid.uuid4().hex[:8]
                    path = (
                        Path(type)
                        / (
                            ""
                            if type in ["random", "random_same_size"]
                            else f"{top_color}_on_top"
                        )
                        / f"red{size_red:.2f}_blue{size_blue:.2f}_{idx}_{unique_hex}.png"
                    )
                    img.save(output_folder / path)
                    writer.writerow(
                        [
                            path,
                            background_color,
                            type,
                            arc_curvature,
                            width,
                            size_red,
                            size_blue,
                            (
                                "none"
                                if type in ["random", "random_same_size"]
                                else top_color
                            ),
                            idx,
                            size_red,
                            size_blue,
                            size_red - size_blue,
                        ]
                    )
    return str(output_folder)


if __name__ == "__main__":
    description = "We used a red and a blue shape arc shape, either one on top of the other at the centre of the canvas (`illusory' and `different lengths' conditions) or randomly placed in the canvas with a random orientation (`scrambled' condition). In the `scrambled' and `different lengths' conditions the two shapes have different sizes. The size is the same (thus eliciting the illusion) in the `illusory' condition. For all conditions, which shape is on top, the shapes' size, the shapes' curvature, and their position and orientation for the `scrambled' condition."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples_illusory",
        "-nis",
        default=DEFAULTS["num_samples_illusory"],
        help="Number of samples for the illusory configuration, with the two objects aligned vertically and with the same size",
        type=int,
    )
    parser.add_argument(
        "--num_samples_aligned",
        "-nas",
        default=DEFAULTS["num_samples_aligned"],
        help="Number of samples for a configuration in which the two objects are vertically aligned, but they don't have the same size",
        type=int,
    )
    parser.add_argument(
        "--num_samples_random",
        "-nrs",
        default=DEFAULTS["num_samples_random"],
        help="Number of samples for condition in which the two objects are randomly placed in the canvas",
        type=int,
    )
    parser.add_argument(
        "--num_samples_random_same_size",
        "-nrss",
        default=DEFAULTS["num_samples_random_same_size"],
        help="Number of samples for condition in which the two objects are randomly placed in the canvas and they are the same size",
        type=int,
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
