"""
This script includes:
1. a colour picker stimuli class that generates colour picker task stimuli. The end goal is to train a general colour picker decoder that can be used for a variety of tasks.
The shapes including:
    - circle
    - chord
    - ellipse
    - pie slice
    - polygon
    - regular polygon
    - rectangle
    - rounded rectangle
2. a shape config class
3. an add arrow function
"""

from PIL import ImageDraw
import PIL.Image as Image

from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
import numpy as np
import math
from random import choice
from pathlib import Path


class ShapeConfigs:
    """
    This class contains the configs for the shapes.
    """

    def __init__(self):
        self.range_area_coords = (0.05, 1)
        self.range_area_circle = (0.05, 1)
        self.threshold_angle_polygon = 45
        self.frequency_ratio = {
            "chords": 1,
            "circles": 1,
            "ellipses": 1,
            "pie_slices": 1,
            "polygons": 0.3,
            "rectangles": 1,
            "regular_polygons": 1,
            "rounded_rectangles": 1,
        }

        sum_values = sum(self.frequency_ratio.values())

        for key in self.frequency_ratio:
            self.frequency_ratio[key] /= sum_values

        self.shape = np.random.choice(
            list(self.frequency_ratio.keys()), p=list(self.frequency_ratio.values())
        )

    def _refresh(self):
        self.shape = np.random.choice(
            list(self.frequency_ratio.keys()), p=list(self.frequency_ratio.values())
        )

    def _return_shape_config(self):
        return {"shape": self.shape, "parameters": getattr(self, self.shape)()}

    def _calculate_area_coords(self, coord_1, coord_2):
        width = np.abs(coord_1[0] - coord_2[0])
        height = np.abs(coord_1[1] - coord_2[1])
        return width * height

    def _calculate_area_circle(self, radius):
        return math.pi * radius**2

    # the angle between three coordinates
    def _calculate_angle(self, c1, c2, c3):
        diff1 = c2 - c1
        diff2 = c3 - c2
        angle = math.atan2(diff2[1], diff2[0]) - math.atan2(diff1[1], diff1[0])
        angle = math.degrees(angle)
        angle += 360 if angle < 0 else 0  # make sure positive
        return angle

    def circles(self):
        """
        circle has the following parameters:
            x, y, r, fill

        x, y: the center of the circle
            (float between 0 and 1, which will be scaled to the image size)
        r: the radius of the circle
            (float between 0 and 1, which will be scaled to the image size)
        fill: the fill color of the circle
            (int: black-white; 0-254)
        """
        x = np.random.rand()
        y = np.random.rand()
        min_radius = (min(self.range_area_circle) / math.pi) ** 0.5
        max_radius = (max(self.range_area_circle) / math.pi) ** 0.5
        r = np.random.uniform(min_radius, max_radius)
        fill = np.random.randint(1, 254)
        return {"x": x, "y": y, "r": r, "fill": fill}

    def chords(self):
        """
        chord has the following parameters:
            coord_1, coord_2, starting_angle, ending_angle, fill

        coord_1, coord_2: the bounding box of the chord
            (float tuples between 0 and 1, which will be scaled to the image size)
        starting_angle: the starting angle of the chord
            (float between 0 and 360)
        ending_angle: the ending angle of the chord
            (float between 0 and 360)
        fill: the fill color of the chord
            (int: black-white; 0-254)
        """
        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        # make sure the difference between the two angles is not too small
        starting_angle, ending_angle = np.random.rand(2) * 360
        while abs(starting_angle - ending_angle) < 30:
            starting_angle, ending_angle = np.random.rand(2) * 360

        fill = np.random.randint(1, 254)
        return {
            "coord_1": coord_1,
            "coord_2": coord_2,
            "starting_angle": starting_angle,
            "ending_angle": ending_angle,
            "fill": fill,
        }

    def ellipses(self):
        """
        ellipse has the following parameters:
            coord_1, coord_2, fill, width

        coord_1, coord_2: the bounding box of the ellipse
            (float tuples between 0 and 1, which will be scaled to the image size)
        fill: the fill color of the ellipse
            (int: black-white; 0-254)
        width: the width of the ellipse
            (float between 0 and 1, which will be scaled to the image size)
        """

        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        width = np.random.uniform(0.3, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        fill = np.random.randint(1, 254)
        width = np.random.rand() / 2
        while width < 0.01:
            width = np.random.rand() / 2
        return {"coord_1": coord_1, "coord_2": coord_2, "fill": fill, "width": width}

    def pie_slices(self):
        """
        pie slice has the following parameters:
            coord_1, coord_2, starting_angle, ending_angle, fill

        coord_1, coord_2: the bounding box of the pie slice
            (float tuples between 0 and 1, which will be scaled to the image size)
        starting_angle: the starting angle of the pie slice
            (float between 0 and 360)
        ending_angle: the ending angle of the pie slice
            (float between 0 and 360)
        fill: the fill color of the pie slice
            (int: black-white; 0-254)
        """

        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        starting_angle, ending_angle = np.random.rand(2) * 360
        while abs(starting_angle - ending_angle) < 30:
            starting_angle, ending_angle = np.random.rand(2) * 360

        fill = np.random.randint(1, 254)
        return {
            "coord_1": coord_1,
            "coord_2": coord_2,
            "starting_angle": starting_angle,
            "ending_angle": ending_angle,
            "fill": fill,
        }

    def polygons(self):
        """
        polygon has the following parameters:
            coordinates, fill
        coordinates: the coordinates of the polygon
            (float tuples between 0 and 1, which will be scaled to the image size)
        fill: the fill color of the polygon
            (int: black-white; 0-254)
        """
        n_points = np.random.randint(3, 10)  # number of points between 3 and 10

        coordinates = np.random.rand(n_points, 2)  # n_points with 2 coordinates
        angles = [
            self._calculate_angle(c1, c2, c3)
            for c1, c2, c3 in zip(coordinates[:-2], coordinates[1:-1], coordinates[2:])
        ]
        while any(angle < self.threshold_angle_polygon for angle in angles):
            coordinates = np.random.rand(n_points, 2)
            angles = [
                self._calculate_angle(c1, c2, c3)
                for c1, c2, c3 in zip(
                    coordinates[:-2], coordinates[1:-1], coordinates[2:]
                )
            ]

        coordinates = coordinates.tolist()
        coordinates = [tuple(coord) for coord in coordinates]

        fill = np.random.randint(1, 254)
        return {"coordinates": coordinates, "fill": fill}

    def regular_polygons(self):
        """
        regular polygon has the following parameters:
            bounding_circle_xyr, n_sides, rotation, fill

        bounding_circle_xyr: the bounding circle of the regular polygon
            (float tuples between 0 and 1, which will be scaled to the image size)
        n_sides: the number of sides of the regular polygon
            (int)
        rotation: the rotation of the regular polygon
            (float between 0 and 360)
        fill: the fill color of the regular polygon
            (int: black-white; 0-254)
        """
        x = np.random.rand()
        y = np.random.rand()
        min_radius = (min(self.range_area_circle) / math.pi) ** 0.5
        max_radius = (max(self.range_area_circle) / math.pi) ** 0.5
        r = np.random.uniform(min_radius, max_radius)

        bounding_circle_xyr = (x, y, r)

        n_sides = np.random.randint(3, 10)
        rotation = np.random.rand() * 360
        fill = np.random.randint(1, 254)
        return {
            "bounding_circle_xyr": bounding_circle_xyr,
            "n_sides": n_sides,
            "rotation": rotation,
            "fill": fill,
        }

    def rectangles(self):
        """
        rectangle has the following parameters:
            coord_1, coord_2, fill

        coord_1, coord_2: the bounding box of the rectangle
            (float tuples between 0 and 1, which will be scaled to the image size)
        fill: the fill color of the rectangle
            (int: black-white; 0-254)
        """

        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        fill = np.random.randint(1, 254)
        return {"coord_1": coord_1, "coord_2": coord_2, "fill": fill}

    def rounded_rectangles(self):
        """
        rounded rectangle has the following parameters:
            coord_1, coord_2, radius, fill
        coord_1, coord_2: the bounding box of the rounded rectangle
            (float tuples between 0 and 1, which will be scaled to the image size)
        radius: the radius of the rounded rectangle
            (float between 0 and 1, which will be scaled to the image size)
        fill: the fill color of the rounded rectangle
            (int: black-white; 0-254)
        """

        target_area_size = np.random.uniform(
            min(self.range_area_coords), max(self.range_area_coords)
        )
        coord_1 = (np.random.uniform(0.1, 1), np.random.uniform(0, 0.9))
        width = np.random.uniform(0.12, 1)
        height = target_area_size / width
        coord_2 = (
            coord_1[0] + width * choice([-1, 1]),
            coord_1[1] + height * choice([-1, 1]),
        )
        coord_1, coord_2 = tuple(coord_1), tuple(coord_2)

        radius = np.random.rand() / 8
        fill = np.random.randint(1, 254)
        return {"coord_1": coord_1, "coord_2": coord_2, "radius": radius, "fill": fill}


class ColorPickerStimuli:
    def __init__(self, canvas_size: tuple = (224, 224)):
        # configs
        assert (
            canvas_size[0] == canvas_size[1]
        ), "the color picker train images has to be squares"
        self.target_image_width = canvas_size[0]
        self.initial_expansion = 1

        self.arrow_line_length = 45  # pixels
        self.triangle_height = 30
        self.triangle_width = 20

        self.indicator_circle_radius = 20  # float between 0 and 1
        self.initial_image_width = self.target_image_width * self.initial_expansion

        self.canvas = new(
            "L", (self.initial_image_width, self.initial_image_width), color=0
        )
        self.draw = Draw(self.canvas)

    def add_circles(self, x, y, r, fill):
        x *= self.initial_image_width
        y *= self.initial_image_width
        r *= self.initial_image_width
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def add_chords(self, coord_1, coord_2, starting_angle, ending_angle, fill):
        # this is for avoiding the new x1 must be larger than x0 error introduced in the new version of PIL
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.chord((coord_1, coord_2), starting_angle, ending_angle, fill=fill)

    def add_ellipses(self, coord_1, coord_2, fill, width):
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)
            coord_2 = (x1, y0)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.ellipse((coord_1, coord_2), fill=fill, width=width)

    def add_pie_slices(self, coord_1, coord_2, starting_angle, ending_angle, fill):
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.pieslice((coord_1, coord_2), starting_angle, ending_angle, fill=fill)

    def add_polygons(self, coordinates, fill):
        coordinates = [
            tuple(map(self._multiply_by_canvas_size, coord)) for coord in coordinates
        ]
        self.draw.polygon(coordinates, fill=fill)

    def add_regular_polygons(self, bounding_circle_xyr, n_sides, rotation, fill):
        bounding_circle_xyr = [
            i * self.initial_image_width for i in bounding_circle_xyr
        ]
        self.draw.regular_polygon(bounding_circle_xyr, n_sides, rotation, fill=fill)

    def add_rectangles(self, coord_1, coord_2, fill):
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        self.draw.rectangle((coord_1, coord_2), fill=fill)

    def add_rounded_rectangles(self, coord_1, coord_2, radius, fill):
        x0, y0 = coord_1
        x1, y1 = coord_2

        if x0 > x1:
            coord_1 = (x1, y0)
            coord_2 = (x0, y1)

        x0, y0 = coord_1
        x1, y1 = coord_2

        if y0 > y1:
            coord_1 = (x0, y1)

        coord_1 = tuple(map(self._multiply_by_canvas_size, coord_1))
        coord_2 = tuple(map(self._multiply_by_canvas_size, coord_2))
        # this conversion is performed to avoid a bug in Pillow
        # https://github.com/python-pillow/Pillow/issues/7149
        coord_1 = tuple(map(int, coord_1))
        coord_2 = tuple(map(int, coord_2))
        radius = self._multiply_by_canvas_size(radius)
        self.draw.rounded_rectangle(xy=(coord_1, coord_2), radius=radius, fill=fill)

    def _add_indicator_circle(self, coord: tuple[float, float]):
        fill = None
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        # move the coord one pixel up to give space for the pixel of interest
        coord = (coord[0], coord[1] - 1)

        self.draw.ellipse(
            (
                coord[0] - self.indicator_circle_radius,
                coord[1] - self.indicator_circle_radius,
                coord[0] + self.indicator_circle_radius,
                coord[1] + self.indicator_circle_radius,
            ),
            fill=fill,
            outline=255,
            width=2,
        )

    def _get_pixel_color(self, coord: tuple[float, float]) -> int:
        """get the color of a pixel in the canvas"""
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        return self.canvas.getpixel(coord)

    def _mutate_pixel_color(self, coord: tuple[float, float], target_color: int):
        """mutate all color of a given pixel in the canvas"""
        pixel_color = self._get_pixel_color(coord)
        # change all pixels of the same color to the target color
        self.canvas.putdata(
            [
                target_color if pixel_color == color else color
                for color in self.canvas.getdata()
            ]
        )

    def _count_colors_withing_circle(self, coord: tuple[float, float]) -> int:
        """get the pixel colors within a circle given its center and radius"""
        coord = tuple(map(self._multiply_by_canvas_size, coord))
        coord_1 = (
            coord[0] - self.indicator_circle_radius,
            coord[1] - self.indicator_circle_radius,
        )
        coord_2 = (
            coord[0] + self.indicator_circle_radius,
            coord[1] + self.indicator_circle_radius,
        )
        return len(self.canvas.crop((*coord_1, *coord_2)).getcolors())

    def _propose_arrow_coord(self) -> tuple[float, float]:
        """
        Propose a coordinate for the arrow
        params:
            - canvas: the canvas to draw on
        returns:
            - coord: the coordinate of the arrow
        constraints:
            - the coordinate must be within the canvas
            - the pixel of interest must not be white
        """
        # random float between 0.1 and 0.9
        coord = np.random.rand(2) * 0.8 + 0.1
        while self._get_pixel_color(coord) == 255:
            coord = np.random.rand(2) * 0.8 + 0.1
        return coord

    def _shrink_and_save(self, save_as: Path):
        self.canvas = self.canvas.resize(
            (self.target_image_width, self.target_image_width),
            resample=Image.Resampling.LANCZOS,
        )
        self.canvas.save(save_as)

    def _check_undrawn_pixels(self):
        """check the number of undrawn pixels in the canvas"""
        return np.sum(np.array(self.canvas) == 0)

    def _check_proportion_dominant_color(self):
        """
        check the proportion of the dominant color in the canvas
        """
        dominant = np.bincount(np.array(self.canvas).flatten()).argmax()
        return np.sum(np.array(self.canvas) == dominant) / self.initial_image_width**2

    def _check_proportion_smallest_segment(self):
        """check the size of the smallest segment in the canvas (in pixels)"""
        segments = self.canvas.getcolors(self.initial_image_width**2)
        segments.sort(key=lambda x: x[0])
        return segments[0][0] / self.initial_image_width**2

    def _check_num_shades(self):
        """check the number of shades of gray in the canvas"""
        return len(np.unique(np.array(self.canvas)))

    def _multiply_by_canvas_size(self, x):
        return x * self.initial_image_width


def add_arrow(canvas, coord: tuple[float, float], fill=255):
    image_width = canvas.size[0]
    resize_ratio = image_width / (4 * 224)

    # configs for the arrow
    arrow_line_length = 45 * resize_ratio
    triangle_height = 30 * resize_ratio
    triangle_width = 20 * resize_ratio
    line_width = int(2 * resize_ratio)

    coord = (coord[0], coord[1] - 1)

    coord_1 = coord
    coord_2 = (coord[0], coord[1] - arrow_line_length)

    coord_triangle_left = (coord[0] - triangle_width, coord[1] - triangle_height)
    coord_triangle_right = (coord[0] + triangle_width, coord[1] - triangle_height)

    draw = ImageDraw.Draw(canvas)
    draw.line((coord_1, coord_2), fill=fill, width=line_width)
    draw.polygon((coord_1, coord_triangle_left, coord_triangle_right), fill=fill)
    return canvas
