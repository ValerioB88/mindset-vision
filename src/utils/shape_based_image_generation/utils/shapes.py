import cv2
import numpy as np
import os
from pathlib import Path


class SizeMonitor:
    def __init__(self, window=3):
        self.window = window
        self.sizes = []

    def add(self, size):
        self.sizes.append(size)

    def check_stable(self):
        # if the last n are the same and different from the initial number
        return (
            len(set(self.sizes[-self.window :])) == 1
            and self.sizes[-1] != self.sizes[0]
        )


def chaikins_corner_cutting(points, num_iterations):
    for _ in range(num_iterations):
        new_points = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            q = ((3 / 4) * p1[0] + (1 / 4) * p2[0], (3 / 4) * p1[1] + (1 / 4) * p2[1])
            r = ((1 / 4) * p1[0] + (3 / 4) * p2[0], (1 / 4) * p1[1] + (3 / 4) * p2[1])

            new_points.extend([q, r])
        points = new_points
    return points


def silhouettes_to_outlines(file_path, width=1):
    """
    Converting images of silhouettes to outlines
    """
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

    # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Dilate the edges to create the outline with the given width
    kernel = np.ones((width, width), np.uint8)
    outline_image = cv2.dilate(edges, kernel)

    return outline_image


if __name__ == "__main__":
    silhouettes_path = Path("private", "baker-1", "Silhouettes")
    silhouettes_files = os.listdir(silhouettes_path)
    silhouettes_files = [i for i in silhouettes_files if not i.startswith(".")]

    # process_files(silhouettes_files, silhouettes_path, outline_width=2)
    output_path = Path("temp")
    output_path.mkdir(exist_ok=True)

    for file in silhouettes_files:
        file_path = silhouettes_path / file
        outline_image = silhouettes_to_outlines(file_path, width=2)

        output_file_path = output_path / file
        cv2.imwrite(str(output_file_path), outline_image)
