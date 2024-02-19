import argparse
import csv
import pathlib

import sty
import toml
import inspect

from tqdm import tqdm
import os

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import DEFAULTS, add_general_args, delete_and_recreate_path
from src.utils.shape_based_image_generation.modules.parent import ParentStimuli
from src.utils.shape_based_image_generation.modules.shapes import Shapes
from src.utils.shape_based_image_generation.utils.parallel import parallel_args
from itertools import product
import random
from pathlib import Path
import numpy as np
import uuid

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


class DrawDecomposition(DrawStimuli):
    def __init__(self, shape_size, shape_color, moving_distance, *args, **kwargs):
        self.shape_size = shape_size
        self.moving_distance = moving_distance
        self.shape_color = shape_color
        super().__init__(*args, **kwargs)

    def generate_canvas(
        self,
        shape_1_name,
        shape_2_name,
        split_type,
        cut_rotation,
        image_rotation=0,
        image_position=(0.5, 0.5),
    ):
        parent = ParentStimuli(
            target_image_size=self.canvas_size,
            initial_expansion=4 if self.antialiasing else 1,
        )

        # create shapes -------------------------------------------
        shape_1 = Shapes(parent)
        shape_2 = Shapes(parent)

        if shape_1_name.split("_")[0] == "puddle":
            shape_1.add_puddle(size=self.shape_size, seed=shape_1_name.split("_")[1])
            shape_2.add_puddle(size=self.shape_size, seed=shape_2_name.split("_")[1])
        else:
            getattr(shape_1, f"add_{shape_1_name}")(**{"size": self.shape_size})
            getattr(shape_2, f"add_{shape_2_name}")(**{"size": self.shape_size})
        shape_1.rotate(30)
        shape_2.rotate(30)

        shape_2.move_next_to(shape_1, "LEFT")

        if split_type == "no_split":
            shape_1.register()
            shape_2.register()

        elif split_type == "unnatural":
            piece_1, piece_2 = shape_2.cut(
                reference_point=(0.5, 0.5), angle_degrees=cut_rotation
            )
            index = np.argmax(
                [piece_1.get_distance_from(shape_1), piece_2.get_distance_from(shape_1)]
            )
            further_piece = [piece_1, piece_2][index]
            closer_piece = [piece_1, piece_2][1 - index]
            further_piece.move_apart_from(closer_piece, self.moving_distance)
            shape_1.register()
            piece_1.register()
            piece_2.register()

        elif split_type == "natural":
            shape_2.move_apart_from(shape_1, self.moving_distance)
            shape_1.register()
            shape_2.register()

        parent.binary_filter()
        parent.convert_color_to_color((255, 255, 255), self.shape_color)
        parent.move_to(image_position).rotate(image_rotation)
        self.create_canvas()  # dummy call to update the background for rnd-uniform mode
        parent.add_background(self.background)
        parent.shrink() if self.antialiasing else None
        return parent.canvas


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "moving_distance": 60,
        "shape_color": [255, 255, 255],
        "output_folder": f"data/{category_folder}/{name_dataset}",
        "number_unfamiliar_shapes": 5,
    }
)


def generate_all(
    moving_distance=DEFAULTS["moving_distance"],
    shape_color=DEFAULTS["shape_color"],
    output_folder=DEFAULTS["output_folder"],
    number_unfamiliar_shapes=DEFAULTS["number_unfamiliar_shapes"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    familiar_shapes = ["arc", "circle", "square", "rectangle", "polygon", "triangle"]
    unfamiliar_shapes = [f"puddle_{i}" for i in range(number_unfamiliar_shapes)]

    combinations_familiar = list(product(familiar_shapes, familiar_shapes))
    combinations_familiar = [
        {
            "shape_1_name": shape_1_name,
            "shape_2_name": shape_2_name,
        }
        for shape_1_name, shape_2_name in combinations_familiar
    ]

    combinations_unfamiliar = list(product(unfamiliar_shapes, unfamiliar_shapes))
    combinations_unfamiliar = [
        {
            "shape_1_name": shape_1_name,
            "shape_2_name": shape_2_name,
        }
        for shape_1_name, shape_2_name in combinations_unfamiliar
    ]

    shapes_types = {
        "familiar": combinations_familiar,
        "unfamiliar": combinations_unfamiliar,
    }
    shape_size = 0.05
    ds = DrawDecomposition(
        shape_size,
        shape_color,
        moving_distance,
        background=background_color,
        canvas_size=canvas_size,
        antialiasing=antialiasing,
    )

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    split_types = ["no_split", "unnatural", "natural"]
    [
        [
            (output_folder / name_comb / split_type).mkdir(exist_ok=True, parents=True)
            for split_type in split_types
        ]
        for name_comb in list(shapes_types.keys())
    ]
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "BackgroundColor",
                "ShapeType",
                "SplitType",
                "LeftShape",
                "RightShape",
                "CutRotation",
                "PairShapeId",
                "IterNum",
            ]
        )
        for name_comb, combs in tqdm(shapes_types.items()):
            for idx, c in enumerate(tqdm(combs, leave=False)):
                cut_rotation = random.uniform(0, 360)
                for split_type in split_types:
                    img = ds.generate_canvas(
                        c["shape_1_name"],
                        c["shape_2_name"],
                        split_type=split_type,
                        cut_rotation=cut_rotation,
                    )
                    unique_hex = uuid.uuid4().hex[:8]
                    path = Path(name_comb) / split_type / f"{unique_hex}.png"
                    img.save(output_folder / path)
                    writer.writerow(
                        [
                            path,
                            ds.background,
                            name_comb,
                            split_type,
                            c["shape_1_name"],
                            c["shape_2_name"],
                            cut_rotation,
                            idx,
                        ]
                    )
    return str(output_folder)


if __name__ == "__main__":
    description = "The dataset consists of images containing simple two parts. There are three `split' conditions and two `familiarity' conditions. The `split' conditions are: `no split` in which two parts are touching at one point but not overlapping; `natural split`, in which two parts are separated; `unnatural split` in which the two parts are touching each other as in the `no split` condition, but one of the parts is `cut' and separated from the rest. The items are silhouettes uniformly coloured on a uniform background, and they can be either familiar or unfamiliar shapes. The familiar shapes consist of the following objects: circle, square, rectangle, triangle, heptagon, and a 50-degree arc segment; the unfamiliar shapes consist of blob-like objects. Within each familiar/unfamiliar condition, all possible combinations of two shapes are used (e.g. a triangle with a rectangle). Configuration parameters include the distance between the pieces in the `unnatural split' and `natural split' condition, the colour of the items, and the number of different blob-like objects to use for the unfamiliar condition."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--moving_distance",
        "-mv",
        default=DEFAULTS["moving_distance"],
        type=int,
        help="Specify by how much each image is separated (same values for the whole dataset)",
    )
    parser.add_argument(
        "--shape_color",
        "-shpc",
        default=DEFAULTS["shape_color"],
        type=lambda x: ([int(i) for i in x.split("_")] if isinstance(x, str) else x),
        help="Specify the color of the shapes (same across the whole dataset). Specify in R_G_B format, e.g. 255_0_0 for red",
    )
    parser.add_argument(
        "--number_unfamiliar_shapes",
        "-nunfs",
        default=DEFAULTS["number_unfamiliar_shapes"],
        type=int,
        help="Specify the number of unfamiliar shapes to use",
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
