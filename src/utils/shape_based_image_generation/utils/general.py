import decimal
import numpy as np
from PIL import Image
from typing import Union, Tuple


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    context = decimal.Context()
    context.pred = 36
    return format(context.create_decimal(repr(f)), "f")


def get_area_size(
    canvas: Image, color: Union[str, Tuple[int, int, int]], tolerance: int = 10
) -> int:
    """
    Get the area size of the given color.
        params:
            canvas: the canvas to get the area size from
            color: the color to get the area size from
        returns:
            area_size: the area size of the given color
    """

    color_dict = {"red": (255, 0, 0), "blue": (0, 0, 255)}

    if type(color) == str:
        assert (
            color in color_dict.keys()
        ), f"color must be one of {list(color_dict.keys())}"
        color = color_dict[color]

    if len(color) == 4:
        color = color[:3]

    canvas_array = np.array(canvas)[:, :, :3]  # ignore the alpha channel

    return np.sum(np.all(np.abs(canvas_array - color) < tolerance, axis=-1))
