"""
This script is for segmenting images from research paper screenshots
as in some research the image shape_based_image_generation is not available as usable format
(strong complaints.......)
"""


import cv2
import numpy as np
from skimage import measure
import os
from pathlib import Path
from rich.traceback import install
from tqdm import tqdm

install()


def segment_image(path_to_image: Path, min_area=50):
    """
    Segments the image into objects and returns a list of numpy arrays
    args:
        path_to_image: path to a folder where the images are stored.
        The images are screenshots of research papers
    returns:
        list of numpy arrays, each representing an object found in the image
        these will include letters, numbers, and symbols
    """
    # Load the image in grayscale
    image = cv2.imread(str(path_to_image), cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_OPEN, kernel)

    labels = measure.label(opened_image, connectivity=2, background=0)

    objects = []

    for label in tqdm(np.unique(labels)):
        if label == 0:  # Skip the background
            continue

        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255

        # Check if the area of the object is greater than the minimum area threshold
        if np.sum(mask == 255) < min_area:
            continue

        # Extract the object and store it as a numpy array
        object_np = cv2.bitwise_and(opened_image, opened_image, mask=mask)
        objects.append(object_np)

    return objects


if __name__ == "__main__":
    PATH_TO_IMAGES = Path("assets", "silhouettes")
    images = os.listdir(PATH_TO_IMAGES)

    objects = [segment_image(PATH_TO_IMAGES / i) for i in images]
    objects = [item for sublist in objects for item in sublist]

    # sort objects by size as we don't want letters
    objects = sorted(objects, key=lambda x: np.sum(x == 255), reverse=True)
    print(f"Found {len(objects)} objects from the images")

    # store all objects as png in the temp folder
    for i, obj in tqdm(enumerate(objects)):
        cv2.imwrite(f"temp/{i}.png", obj)
