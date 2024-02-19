"""
The code in this page is copied from the GPL 3.0 licensed GitHub repo:
https://github.com/Erfaniaa/thatcher-effect-dataset-generator/tree/master
The function get_image_facial_landmark has been rewritten to get rid of the dependency from the (heavy and error-prone) dlib. main() function has been removed.
Notice that this requires the installation of opencv-contrib: pip install opencv-contrib-python --user
"""

import numpy as np
import matplotlib.pyplot as plt
from math import inf
import cv2


def get_image_facial_landmarks(image_path, facemark):
    face_cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    ret = []

    image = plt.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        ok, landmarks = facemark.fit(gray, faces)

        if ok:
            for landmark in landmarks:
                for x, y in np.squeeze(landmark):
                    ret.append((x, y))

    return ret


def get_bounding_rectangle(points):
    top_left = [inf, inf]
    bottom_right = [-inf, -inf]
    for point in points:
        top_left[0] = min(top_left[0], point[1])
        top_left[1] = min(top_left[1], point[0])
        bottom_right[0] = max(bottom_right[0], point[1])
        bottom_right[1] = max(bottom_right[1], point[0])
    return [top_left, bottom_right]


def flip_subimage_vertically(image, x1, y1, x2, y2):
    mid_x = (x1 + x2) // 2
    for x in range(x1, mid_x):
        for y in range(y1, y2 + 1):
            image[x][y], image[x1 + x2 - x][y] = (
                image[x1 + x2 - x][y].copy(),
                image[x][y].copy(),
            )


def flip_subimage_ellipse_vertically(image, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1 and x1 + x2 - x > x:
                image[x][y], image[x1 + x2 - x][y] = (
                    image[x1 + x2 - x][y].copy(),
                    image[x][y].copy(),
                )


def gradient_subimage(image, x1, y1, x2, y2):
    final_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
    start_color = image[x1][y1].copy()
    final_color = image[x2][y2].copy()
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            current_distance = (x - x1) ** 2 + (y - y1) ** 2
            k = current_distance / final_distance
            current_color = start_color * (1 - k) + final_color * k
            image[x][y] = current_color


def blur_ellipse_border(image, x1, y1, x2, y2):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.25 and (dx * dx) / (
                a * a
            ) + (dy * dy) / (b * b) >= 0.75:
                image[x][y] = blurred_image[x][y]


def blur_orthogonal_border(image, blurred_image, x1, y1, x2, y2, border_size):
    if x1 == x2:
        for x in range(x1 - border_size, x1 + border_size + 1):
            for y in range(y1, y2 + 1):
                image[x][y] = blurred_image[x][y]
    if y1 == y2:
        for y in range(y1 - border_size, y1 + border_size + 1):
            for x in range(x1, x2 + 1):
                image[x][y] = blurred_image[x][y]


def blur_rectangle_border(image, x1, y1, x2, y2, border_size=2):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    blur_orthogonal_border(image, blurred_image, x1, y1, x2, y1, border_size)
    blur_orthogonal_border(image, blurred_image, x1, y2, x2, y2, border_size)
    blur_orthogonal_border(image, blurred_image, x1, y1, x1, y2, border_size)
    blur_orthogonal_border(image, blurred_image, x2, y1, x2, y2, border_size)


def flip_subimage_vertically_with_border_softening(image, x1, y1, x2, y2):
    flip_subimage_vertically(image, x1, y1, x2, y2)
    blur_rectangle_border(image, x1, y1, x2, y2)


def flip_subimage_ellipse_vertically_with_border_softening(image, x1, y1, x2, y2):
    flip_subimage_ellipse_vertically(image, x1, y1, x2, y2)
    blur_ellipse_border(image, x1, y1, x2, y2)


def apply_thatcher_effect_on_image(
    input_image_path,
    left_eye_rectangle,
    right_eye_rectangle,
    mouth_rectangle,
):
    image = cv2.cvtColor(
        cv2.imread(
            input_image_path,
        ),
        cv2.COLOR_BGR2RGB,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        left_eye_rectangle[0][0] - 5,
        left_eye_rectangle[0][1] - 6,
        left_eye_rectangle[1][0] + 7,
        left_eye_rectangle[1][1] + 3,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        right_eye_rectangle[0][0] - 5,
        right_eye_rectangle[0][1] - 3,
        right_eye_rectangle[1][0] + 7,
        right_eye_rectangle[1][1] + 6,
    )
    flip_subimage_ellipse_vertically_with_border_softening(
        image,
        mouth_rectangle[0][0] - 4,
        mouth_rectangle[0][1] - 5,
        mouth_rectangle[1][0] + 3,
        mouth_rectangle[1][1] + 5,
    )
    return image
