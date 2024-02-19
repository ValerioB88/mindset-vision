import argparse
import os
import csv
import pathlib
import random
import numpy as np
import cv2
from PIL import Image
import uuid

import toml
import inspect


from .utils import (
    get_line_points,
    sample_midpoints_lines,
    svrt_1_points,
)
import sty
from tqdm import tqdm
from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import (
    add_general_args,
    apply_antialiasing,
    delete_and_recreate_path,
)

from src.utils.misc import DEFAULTS as BASE_DEFAULTS

DEFAULTS = BASE_DEFAULTS.copy()


class DrawSameDifferentStimuli(DrawStimuli):
    def svrt_1_img(
        self,
        category=1,
        size1=None,
        size2=None,
        regular=None,
        rotations=None,
        sides=None,
        irregularity=0.5,
        thickness=1,
        color_a=None,
        color_b=None,
        filled=False,
        closed=True,
    ):
        """Returns a picture of single instance of a SVRT problem 1.
        Args:
            category: 0 (no) or 1 (yes).
            radii: radii of the base regular polygon. 2-tuple 8 to 14.
            sides: number of sides of the base regular polygon. 2-tuple 4 to 8.
            rotations: rotations of the polygons. 2-tuple 4 to 8.
            regular: whether to build regular or irregular polygons in radiants. 2-tuple form 0 to pi.
            irregularity: maximum level of random point translation for irregular polygons.
            thickness: line width of the shapes.
            color: line color of the shapes.
        Returns:
            Numpy array."""

        img = np.array(self.create_canvas())
        color_a = self.fill if color_a is None else color_a

        if color_b is None:
            color_b = color_a

        points_a, points_b, _, _ = svrt_1_points(
            category=category,
            radii=(size1, size2),
            sides=sides,
            rotations=rotations,
            regular=regular,
            irregularity=irregularity,
            canvas_size=self.canvas_size,
        )

        poly_a = np.array(points_a, dtype=np.int32)
        poly_b = np.array(points_b, dtype=np.int32)

        poly_new_a = poly_a.reshape((-1, 1, 2))
        poly_new_b = poly_b.reshape((-1, 1, 2))

        # Draw.
        if not filled:
            cv2.polylines(
                img, [poly_new_a], isClosed=closed, color=color_a, thickness=thickness
            )

            cv2.polylines(
                img,
                [poly_new_b],
                isClosed=closed,
                color=color_b,
                thickness=thickness,
            )

        else:
            cv2.fillPoly(img, [poly_new_a], color=color_a)
            cv2.fillPoly(img, [poly_new_b], color=color_b)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_straight_lines_sd_diffrot(self, category, size1, size2, line_thickness=1):
        img = np.array(self.create_canvas())

        rotations = random.sample([0, 45, 90, 135], 2)
        rotation_1 = rotations[0]
        rotation_2 = rotations[1]

        if category == 1:
            rotation_2 = rotation_1

        midpoint_1, midpoint_2 = sample_midpoints_lines(
            sizes=(size1, size2), canvas_size=self.canvas_size
        )

        points_line_1 = get_line_points(
            size=size1, rotation=rotation_1, center=midpoint_1
        )
        points_line_2 = get_line_points(
            size=size2, rotation=rotation_2, center=midpoint_2
        )

        cv2.line(
            img, points_line_1[0], points_line_1[1], self.fill, thickness=line_thickness
        )
        cv2.line(
            img,
            points_line_2[0],
            points_line_2[1],
            self.fill,
            thickness=line_thickness,
        )

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_squares_sd(self, size1, size2, category):
        img = np.array(self.create_canvas())

        if category == 1:
            size2 = size1

        x_1 = random.sample(list(range(2, self.canvas_size[0] - (size1 + 2))), 1)[0]
        y_1 = random.sample(list(range(2, self.canvas_size[1] - (size1 + 2))), 1)[0]
        x_2 = random.sample(list(range(2, self.canvas_size[0] - (size2 + 2))), 1)[0]
        y_2 = random.sample(list(range(2, self.canvas_size[1] - (size2 + 2))), 1)[0]
        start_point_1 = (x_1, y_1)
        start_point_2 = (x_2, y_2)
        end_point_1 = (x_1 + size1, y_1 + size1)
        end_point_2 = (x_2 + size2, y_2 + size2)

        img = cv2.rectangle(img, start_point_1, end_point_1, self.fill, 1)
        img = cv2.rectangle(img, start_point_2, end_point_2, self.fill, 1)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_rectangles_sd(self, size1, category):
        img = np.array(self.create_canvas())
        const_dim = "x" if random.random() > 0.5 else "y"

        if const_dim == "y":
            size_x_1 = size1
            size_x_2 = (
                random.sample([size1 - size1 // 2, size1 + size1 // 2], 1)[0]
                if category == 0
                else size_x_1
            )

            size_y_1 = size1
            size_y_2 = size_y_1

        elif const_dim == "x":
            size_y_1 = size1
            size_y_2 = (
                random.sample([size1 - size1 // 2, size1 + size1 // 2], 1)[0]
                if category == 0
                else size_y_1
            )

            size_x_1 = size1
            size_x_2 = size_x_1

        # Sample start and end points.
        x_1 = random.sample(list(range(2, self.canvas_size[0] - (size_x_1 + 2))), 1)[0]
        y_1 = random.sample(list(range(2, self.canvas_size[1] - (size_y_1 + 2))), 1)[0]
        x_2 = random.sample(list(range(2, self.canvas_size[0] - (size_x_2 + 2))), 1)[0]
        y_2 = random.sample(list(range(2, self.canvas_size[1] - (size_y_2 + 2))), 1)[0]
        start_point_1 = (x_1, y_1)
        start_point_2 = (x_2, y_2)
        end_point_1 = (x_1 + size_x_1, y_1 + size_y_1)
        end_point_2 = (x_2 + size_x_2, y_2 + size_y_2)

        # Draw squares.
        img = cv2.rectangle(img, start_point_1, end_point_1, self.fill, 1)
        img = cv2.rectangle(img, start_point_2, end_point_2, self.fill, 1)

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img

    def make_connected_open_squares(
        self, size1, category, line_width=1, is_closed=False
    ):
        img = np.array(self.create_canvas())

        size = size1
        points_a = [
            [0, size],
            [0, 0],
            [size, 0],
            [size, size],
            [size, 2 * size],
            [2 * size, 2 * size],
            [2 * size, size],
        ]

        points_b = [
            [0, size],
            [0, 2 * size],
            [size, 2 * size],
            [size, size],
            [size, 0],
            [2 * size, 0],
            [2 * size, size],
        ]
        # Assign points based on category.
        if category == 1:
            points_b = points_a

        # Sample translations and apply.
        translation_a = [
            np.random.randint(1, self.canvas_size[0] - size * 2),
            np.random.randint(1, self.canvas_size[0] - size * 2),
        ]
        translation_b = [
            np.random.randint(1, self.canvas_size[0] - size * 2),
            np.random.randint(1, self.canvas_size[0] - size * 2),
        ]
        points_a = [
            [sum(pair) for pair in zip(point, translation_a)] for point in points_a
        ]
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in points_b
        ]

        poly_a = np.array(points_a, dtype=np.int32)
        poly_b = np.array(points_b, dtype=np.int32)

        poly_new_a = poly_a.reshape((-1, 1, 2))
        poly_new_b = poly_b.reshape((-1, 1, 2))

        cv2.polylines(
            img, [poly_new_a], isClosed=is_closed, color=self.fill, thickness=line_width
        )
        cv2.polylines(
            img,
            [poly_new_b],
            isClosed=is_closed,
            color=self.fill,
            thickness=line_width,
        )

        img = Image.fromarray(img)
        return apply_antialiasing(img) if self.antialiasing else img


def is_overlapping(img: np.array, background_color: tuple, threshold: int = 2):
    img_c = img.copy()
    img_c[img == background_color] = 0

    # diff_img = cv2.absdiff(img, background_img)
    gray = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = (
        cnts[0] if len(cnts) == 2 else cnts[1]
    )  # this is to deal with different opencv versions.

    b_rects = []
    for c in cnts:
        b_rects.append(cv2.boundingRect(c))

    return len(b_rects) != 2


get_irregular_polygon = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, **kwargs
)

get_regular = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=True, sides=None, thickness=1, **kwargs
)


get_open = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label,
    regular=False,
    sides=None,
    thickness=1,
    closed=False,
    **kwargs,
)

get_wider_line = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=2, **kwargs
)


def get_rnd_color(ds, label, **kwargs):
    color = tuple(np.random.randint(1, high=256, size=3))
    color = (int(color[0]), int(color[1]), int(color[2]))
    img = ds.svrt_1_img(
        category=label, regular=False, color_a=color, sides=None, thickness=1, **kwargs
    )
    return img


get_filled = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label,
    regular=False,
    sides=None,
    thickness=1,
    filled=True,
    **kwargs,
)

get_arrows = lambda ds, label, **kwargs: ds.make_arrows_sd(
    category=label, continuous=True, line_width=1, hard_test=True, **kwargs
)

get_straight_lines = lambda ds, label, **kwargs: ds.make_straight_lines_sd_diffrot(
    category=label, line_thickness=1, **kwargs
)

get_rectangles = lambda ds, label, **kwargs: ds.make_rectangles_sd(
    category=label, **kwargs
)

get_open_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, **kwargs
)

get_closed_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, is_closed=True, **kwargs
)


DEFAULTS.update({"num_samples": 5000, "type_dataset": "all", "size_shapes": 20})


def is_integer(n):
    try:
        int(n)
        return True
    except ValueError:
        return False


category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))
DEFAULTS["output_folder"] = f"data/{category_folder}/{name_dataset}"


def generate_all(
    size_shapes=DEFAULTS["size_shapes"],
    num_samples=DEFAULTS["num_samples"],
    type_dataset=DEFAULTS["type_dataset"],
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

    datasets = {
        "regular": get_regular,
        "irregular": get_irregular_polygon,
        "open": get_open,
        "wider_line": get_wider_line,
        "rnd_color": get_rnd_color,
        "filled": get_filled,
        "open_squares": get_open_squares,
        "rectangles": get_rectangles,
        "straight_lines": get_straight_lines,
        "closed_squares": get_closed_squares,
    }

    ds = DrawSameDifferentStimuli(
        background=background_color, canvas_size=canvas_size, antialiasing=antialiasing
    )

    if not type_dataset == "all":
        datasets = datasets[type_dataset]

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    labels = ["same", "diff"]
    [
        [
            (output_folder / ds / l).mkdir(exist_ok=True, parents=True)
            for ds in datasets.keys()
        ]
        for l in labels
    ]

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "BackgroundColor",
                "TypeDataset",
                "SizeShape1",
                "SizeShape2",
                "SameDiff",
                "SampleNum",
            ]
        )
        for ds_name, dataset_fun in tqdm(datasets.items()):
            use_size2 = True
            if ds_name in [
                "open_squares",
                "closed_squares",
                "rectangles",
            ]:
                use_size2 = False
            for n in tqdm(range(num_samples), leave=False):
                for label in labels:
                    while True:
                        # print(f"\n{ds_name} label: {label}")
                        if is_integer(size_shapes):
                            size1, size2 = size_shapes, size_shapes
                        else:
                            if size_shapes == "rnd1":
                                size1 = np.random.randint(
                                    ds.canvas_size[0] // 15, ds.canvas_size[0] // 4
                                )
                                size2 = size1
                            elif size_shapes == "rnd2":
                                size1, size2 = np.random.randint(
                                    ds.canvas_size[0] // 15, ds.canvas_size[0] // 4, 2
                                )

                        args = dict(label=1 if label == "same" else 0, size1=size1)
                        args.update({"size2": size2}) if use_size2 else None

                        img = dataset_fun(ds, **args)

                        if is_overlapping(np.array(img), ds.background):
                            pass
                        else:
                            break
                    unique_hex = uuid.uuid4().hex[:8]
                    img_path = pathlib.Path(ds_name) / label / f"{unique_hex}.png"
                    img.save(output_folder / img_path)
                    writer.writerow(
                        [
                            img_path,
                            background_color,
                            ds_name,
                            size1,
                            size2 if use_size2 else None,
                            label,
                            n,
                        ]
                    )
    return str(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])

    parser.add_argument(
        "--num_samples",
        "-nis",
        default=DEFAULTS["num_samples"],
        help="Number of generated samples for each type of dataset and each same/different condition",
        type=int,
    )

    parser.add_argument(
        "--size_shapes",
        "-ss",
        default=DEFAULTS["size_shapes"],
        type=str,
        help="either a number (both shapes the same, specific size), or rnd1 (different random sizes across samples, but same size within samples), rnd2 (different random sizes within samples and across samples). Different sizes for each shape are only applied when it makes sense for the task.",
    )
    parser.add_argument(
        "--type_dataset",
        "-nas",
        default=DEFAULTS["type_dataset"],
        help="Specify the type of datasets. It could be `all` or any of `regular`, `irregular`, `open`, `wider_line`, `rnd_color`, `filled`, `open_squares`, `rectangles`, `straight_lines`, `closed_squares`",
    )
    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
