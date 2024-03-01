import random
import numpy as np
from PIL import ImageDraw

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import apply_antialiasing


def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = (max(x_coords) + min(x_coords)) / 2
    centroid_y = (max(y_coords) + min(y_coords)) / 2
    return centroid_x, centroid_y


def extend_line(line, factor):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    x2_new, y2_new = x1 + dx * factor, y1 + dy * factor
    x1_new, y1_new = x1 - dx * (factor - 1), y1 - dy * (factor - 1)
    return (x1_new, y1_new, x2_new, y2_new)


def center_and_scale(points, canvas_size, shape_size):
    scaled_points = [
        (
            p[0]
            * shape_size
            / (max([p[0] for p in points]) - min([p[0] for p in points])),
            p[1]
            * shape_size
            / (max([p[1] for p in points]) - min([p[1] for p in points])),
        )
        for p in points
    ]
    centroid = calculate_centroid(scaled_points)

    translated_points = [
        (
            p[0] + canvas_size[0] / 2 - centroid[0],
            p[1] + canvas_size[1] / 2 - centroid[1],
        )
        for p in scaled_points
    ]

    return translated_points


def shift_line(line, width, height, max_shift):
    x1, y1, x2, y2 = line

    shift_x = random.uniform(-max_shift, max_shift)
    shift_y = random.uniform(-max_shift, max_shift)

    # Shift the line
    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    # Ensure the line is still within the canvas
    x1 = min(max(x1, 0), width)
    x2 = min(max(x2, 0), width)
    y1 = min(max(y1, 0), height)
    y2 = min(max(y2, 0), height)

    return x1, y1, x2, y2


def draw_number_exclude_range(min, max, not_min, not_max):
    while True:
        num = random.randint(min, max)
        if num < not_min or num > not_max:
            return num


class DrawEmbeddedFigures(DrawStimuli):
    def __init__(self, shape_size, *args, **kwargs):
        self.shape_size = shape_size
        super().__init__(*args, **kwargs)

    def draw_shape(
        self,
        original_points,
        extend_lines=False,
        num_shift_lines=5,
        num_rnd_lines=0,
    ):
        original_canvas_size = self.canvas_size
        self.canvas_size = tuple(np.array(self.canvas_size))
        canvas = self.create_canvas()
        draw = ImageDraw.Draw(canvas)
        points = center_and_scale(original_points, self.canvas_size, self.shape_size)
        points = [tuple(np.round(np.array(i)).astype(int)) for i in points]
        width, height = canvas.size

        for i in range(len(points) - 1):
            line = points[i] + points[i + 1]
            line = extend_line(line, self.canvas_size[0]) if extend_lines else line
            draw.line(line, **self.line_args)

        for i in range(num_shift_lines):
            i = random.choice(range(len(points) - 1))
            line = shift_line(
                points[i] + points[i + 1],
                width,
                height,
                max_shift=max((width // 2, height // 2)),
            )
            draw.line(
                extend_line(line, 100) if extend_lines else line, **self.line_args
            )

        for _ in range(num_rnd_lines):
            if random.random() < 0.5:
                # Line crosses the shorter dimension of the canvas and exits through the longer one
                x1, y1 = random.random() * width, 0
                x2, y2 = random.random() * width, height
            else:
                # Line crosses the longer dimension of the canvas and exits through the shorter one
                x1, y1 = 0, random.random() * height
                x2, y2 = width, random.random() * height
            draw.line((x1, y1, x2, y2), **self.line_args)

        self.canvas_size = original_canvas_size
        return apply_antialiasing(canvas) if self.antialiasing else canvas
