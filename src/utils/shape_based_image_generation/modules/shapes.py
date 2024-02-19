import math
from src.utils.shape_based_image_generation.utils.shapes import chaikins_corner_cutting
from perlin_noise import PerlinNoise
import random
from src.utils.shape_based_image_generation.modules.core import ShapeCoreFunctions
from PIL.Image import new
from PIL.ImageDraw import Draw
import numpy as np
from PIL import Image
from pathlib import Path


class Shapes(ShapeCoreFunctions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # shape section ------------------------------------------------------------
    def add_arc(self, size, x=0.5, y=0.5, arc=50, width=0.3):
        self.shape = "arc"  # for the jastrow_illusion illusion

        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]
        size *= self.initial_image_size[0] * self.initial_image_size[1]
        width *= self.initial_image_size[0]
        width = max(width, 1).__int__()

        r_outer = (size / (arc * math.pi / 360) + width**2) / (2 * width)
        r_inner = r_outer - width
        center_x = x
        center_y = y + (width / 2) + r_inner
        center = (center_x, center_y)
        start = -90 - arc / 2
        end = -90 + arc / 2

        canvas_inner = new("RGBA", self.initial_image_size, (0, 0, 0, 0))
        canvas_outer = new("RGBA", self.initial_image_size, (0, 0, 0, 0))
        draw_inner = Draw(canvas_inner)
        draw_outer = Draw(canvas_outer)
        x0_inner = min(center[0] - r_inner, center[0] + r_inner)
        y0_inner = min(center[1] - r_inner, center[1] + r_inner)
        x1_inner = max(center[0] - r_inner, center[0] + r_inner)
        y1_inner = max(center[1] - r_inner, center[1] + r_inner)
        x0_outer = min(center[0] - r_outer, center[0] + r_outer)
        y0_outer = min(center[1] - r_outer, center[1] + r_outer)
        x1_outer = max(center[0] - r_outer, center[0] + r_outer)
        y1_outer = max(center[1] - r_outer, center[1] + r_outer)

        draw_inner.ellipse(
            (x0_inner, y0_inner, x1_inner, y1_inner), fill=self.color, width=1
        )
        draw_outer.pieslice(
            (x0_outer, y0_outer, x1_outer, y1_outer),
            start,
            end,
            fill=self.color,
            width=1,
        )

        canvas_inner = np.array(canvas_inner)
        canvas_outer = np.array(canvas_outer)
        canvas = canvas_outer * (1 - canvas_inner[:, :, 3:4] / 255)
        canvas = Image.fromarray(canvas.astype(np.uint8))
        self.canvas.paste(canvas, (0, 0), canvas)
        return self

    def add_circle(self, size, x=0.5, y=0.5):
        radius = math.sqrt(
            size * self.initial_image_size[0] * self.initial_image_size[1] / math.pi
        )
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]
        self.draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=self.color
        )
        return self

    def add_square(self, size, x=0.5, y=0.5):
        side = math.sqrt(size * self.initial_image_size[0] * self.initial_image_size[1])
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]
        half_side = side / 2

        # Compute the coordinates of the unrotated vertices
        top_left = (x - half_side, y + half_side)
        top_right = (x + half_side, y + half_side)
        bottom_left = (x - half_side, y - half_side)
        bottom_right = (x + half_side, y - half_side)

        # Rotate the vertices by the given angle
        vertices = [top_left, top_right, bottom_right, bottom_left]

        # Draw the rotated square
        self.draw.polygon(vertices, fill=self.color)
        return self

    def add_rectangle(self, size, x=0.5, y=0.5, t=2):
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]

        # Calculate the width and height of the rectangle based on the area and side ratio
        size = size * self.initial_image_size[0] * self.initial_image_size[1]
        width = (size / t) ** 0.5
        height = t * width

        half_width = width / 2
        half_height = height / 2

        # Compute the coordinates of the unrotated vertices
        top_left = (x - half_width, y + half_height)
        top_right = (x + half_width, y + half_height)
        bottom_left = (x - half_width, y - half_height)
        bottom_right = (x + half_width, y - half_height)

        # Rotate the vertices by the given angle
        vertices = [top_left, top_right, bottom_right, bottom_left]

        # Draw the rotated rectangle
        self.draw.polygon(vertices, fill=self.color)
        return self

    def add_polygon(self, size, x=0.5, y=0.5, n_sides=7):
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]

        # Calculate the area of the polygon based on the area ratio
        size = size * self.initial_image_size[0] * self.initial_image_size[1]

        # Calculate the radius of the polygon's bounding circle
        radius = math.sqrt((2 * size) / (n_sides * math.sin(2 * math.pi / n_sides)))

        # Create the regular polygon with the calculated radius
        bounding_circle_xyr = [x, y, radius]
        self.draw.regular_polygon(bounding_circle_xyr, n_sides, 0, fill=self.color)
        return self

    def add_triangle(self, size, x=0.5, y=0.5):
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]

        # Calculate the area of the triangle based on the area ratio
        size = size * self.initial_image_size[0] * self.initial_image_size[1]

        # Calculate the side length of the equilateral triangle
        side_length = math.sqrt((4 * size) / math.sqrt(3))

        # Calculate the height and half the base of the triangle
        height = math.sqrt(3) / 2 * side_length
        half_base = side_length / 2

        # Coordinates of the vertices
        vertex_1 = (x, y - height / 3 * 2)
        vertex_2 = (x - half_base, y + height / 3)
        vertex_3 = (x + half_base, y + height / 3)

        self.draw.polygon((vertex_1, vertex_2, vertex_3), fill=self.color)
        return self

    def add_puddle(self, size, x=0.5, y=0.5, seed=1, smooth_iterations=5):
        current_seed = random.getstate()
        x *= self.initial_image_size[0]
        y *= self.initial_image_size[1]

        # Calculate the radius of a circle with the same area as the expected puddle area
        size *= self.initial_image_size[0] * self.initial_image_size[1]
        size *= math.pi  # somewhat works
        circle_radius = (size / math.pi) ** 0.5

        if seed is not None:
            random.seed(seed)

        scale = 0.05
        octaves = 4

        noise = PerlinNoise(
            octaves=octaves, seed=int(seed) if seed is not None else None
        )

        vertices = []
        num_points = 24

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            noise_value = noise(
                [x * scale * math.cos(angle), y * scale * math.sin(angle)]
            )
            noise_radius = circle_radius * (0.5 + 0.5 * noise_value)
            xi = x + noise_radius * math.cos(angle)
            yi = y + noise_radius * math.sin(angle)
            vertices.append((xi, yi))

        # Apply Chaikin's corner cutting algorithm for smoother edges
        vertices = chaikins_corner_cutting(vertices, smooth_iterations)

        self.draw.polygon(vertices, fill=self.color)
        random.seed(current_seed)
        return self


if __name__ == "__main__":
    pass
