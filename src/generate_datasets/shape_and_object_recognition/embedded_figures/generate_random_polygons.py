from math import cos, sin, tau
import random
import numpy as np
from PIL import Image, ImageDraw

"""
This function can be used in conjunction with generate_dataset, but only "manually". In practice, you generate polygons, until they satisfy your criteria, judge by inspection. When a "nice" polygon is generated, you can just copy the vertices values and use them in the "generate_dataset_lines.py" in this folder. Just literally copy and paste them in the code. Obviously, this step is already done and 5 polygons are already present in "generate_dataset_lines.py", but in case you want to change the polygons, here you go.  

Note: these functions do not guarantee that the polygon will be simple (e.g. without intersection). Also the angles will sometime be similar to each other so that it's difficult to see whether there are really the desired number of vertices. 
"""


def generate_random_polygon(n_vertices, canvas_size=100):
    vertices = []
    eps = 0.1
    angle = random.uniform(0, tau)

    angles = [angle]
    for _ in range(1, n_vertices):
        angle = (angle + eps + random.uniform(0, tau - eps)) % tau
        angles.append(angle)
    angles.sort()
    print(angles)

    for angle in angles:
        radius = random.uniform(0, canvas_size / 2)
        x = canvas_size / 2 + radius * cos(angle)
        y = canvas_size / 2 + radius * sin(angle)
        vertices.append((x, y))

    vertices.append(vertices[0])

    return vertices


def change_range(x, initial_range, finale_range):
    scale_factor = (finale_range[1] - finale_range[0]) / (
        initial_range[1] - initial_range[0]
    )

    transformed_x = (x - initial_range[0]) * scale_factor + finale_range[0]

    return transformed_x


img = Image.new("RGB", (100, 100), "white")
draw = ImageDraw.Draw(img)

vertices = generate_random_polygon(7)
rescaled_vertices = [
    tuple(change_range(x, [np.min(vertices), np.max(vertices)], [0, 100]))
    for x in vertices
]
draw.line(rescaled_vertices, width=1, fill="black")
img.show()
