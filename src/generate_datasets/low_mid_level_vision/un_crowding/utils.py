import matplotlib
import numpy as np
from PIL import Image, ImageDraw


def circle_perimeter(r, c, radius, method="bresenham", shape=None):
    """Generate circle perimeter coordinates using Bresenham's algorithm.

    Parameters:
    - r, c: Center coordinates of the circle.
    - radius: Radius of the circle.
    - method: Circle drawing method. Currently, only Bresenham's method is implemented.
    - shape: Image shape for clipping the coordinates.

    Returns:
    - rr, cc: Arrays of row and column coordinates of the circle's perimeter.
    """
    rr, cc = [], []
    x = radius
    y = 0
    d = 3 - 2 * radius

    while y <= x:
        points = [
            (x + r, y + c),
            (-x + r, y + c),
            (x + r, -y + c),
            (-x + r, -y + c),
            (y + r, x + c),
            (-y + r, x + c),
            (y + r, -x + c),
            (-y + r, -x + c),
        ]

        for point in points:
            rr.append(point[0])
            cc.append(point[1])

        if d <= 0:
            d = d + 4 * y + 6
        else:
            d = d + 4 * (y - x) + 10
            x -= 1
        y += 1

    rr, cc = np.array(rr), np.array(cc)

    # Clip coordinates to fit within the specified shape
    if shape is not None:
        valid_idx = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[valid_idx], cc[valid_idx]

    return rr, cc


def polygon(r, c, shape=None):
    """Generate coordinates of pixels within a polygon.

    Parameters:
    ----------
    r : (N,) ndarray
        Row coordinates of vertices of the polygon.
    c : (N,) ndarray
        Column coordinates of vertices of the polygon.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel coordinates.

    Returns:
    -------
    rr, cc : ndarray of int
        Pixel coordinates of the polygon.
    """
    r, c = np.array(r), np.array(c)
    # Create a path from the polygon vertices
    poly_path = matplotlib.path.Path(np.column_stack((c, r)))

    # Create a grid of points to test
    if shape is None:
        x, y = np.meshgrid(
            np.arange(c.min(), c.max() + 1), np.arange(r.min(), r.max() + 1)
        )
    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    points = np.vstack((x.ravel(), y.ravel())).T

    # Use the path to test which points are inside the polygon
    grid = poly_path.contains_points(points)
    inside_points = points[grid]

    # Extract row and column coordinates
    rr, cc = inside_points[:, 1], inside_points[:, 0]

    return rr.astype(int), cc.astype(int)


def line_bresenham(r0, c0, r1, c1):
    """Generate line coordinates using Bresenham's algorithm."""
    width = max(c0, c1) + 1
    height = max(r0, r1) + 1
    img = Image.new("L", (width, height), 0)  # Create a blank image
    draw = ImageDraw.Draw(img)

    # Draw the line on the image
    draw.line((c0, r0, c1, r1), fill=255)

    # Extract the line coordinates
    line_coords = np.array(img).nonzero()
    rr, cc = line_coords

    return rr, cc


def polygon_perimeter(r, c, shape=None, clip=False):
    """Generate polygon perimeter coordinates using Bresenham's line algorithm."""
    if len(r) != len(c):
        raise ValueError("Row and column coordinates must be of the same length.")

    rr_total, cc_total = np.array([], dtype=int), np.array([], dtype=int)
    for i in range(len(r)):
        r0, c0 = r[i], c[i]
        r1, c1 = r[(i + 1) % len(r)], c[(i + 1) % len(r)]
        rr, cc = line_bresenham(
            int(np.round(r0)), int(np.round(c0)), int(np.round(r1)), int(np.round(c1))
        )
        rr_total = np.concatenate((rr_total, rr))
        cc_total = np.concatenate((cc_total, cc))

    if clip and shape is not None:
        rr_total = np.clip(rr_total, 0, shape[0] - 1)
        cc_total = np.clip(cc_total, 0, shape[1] - 1)

    return rr_total, cc_total


def line(r0, c0, r1, c1):
    """Generate line pixel coordinates using the Bresenham algorithm."""
    rr, cc = [], []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    err = dr - dc if dr > dc else dc - dr

    r_step = 1 if r0 < r1 else -1
    c_step = 1 if c0 < c1 else -1

    while True:
        rr.append(r0)
        cc.append(c0)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += r_step
        if e2 < dr:
            err += dr
            c0 += c_step

    return np.array(rr), np.array(cc)
