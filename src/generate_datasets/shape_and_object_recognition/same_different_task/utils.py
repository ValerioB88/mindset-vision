import copy
import math
import random
import numpy as np


def svrt_1_points(
    canvas_size,
    category=1,
    radii=None,
    sides=None,
    rotations=None,
    regular=False,
    irregularity=0.25,
):
    """Returns polygon points for a single instance of a SVRT problem 1.
    Args:
        category: 0 (no) or 1 (yes).
        radii: radii of the base regular polygon. 2-tuple 8 to 14.
        sides: number of sides of the base regular polygon. 2-tuple 4 to 8.
        rotations: rotations of the polygons. 2-tuple 4 to 8.
        regular: whether to build regular or irregular polygons in radiants. 2-tuple form 0 to pi.
        irregularity: maximum level of random point translation for irregular polygons.
        displace_vertices: if True displaces second polygon subseccions randomly around its center in the positive cases.
    Returns:
        Two lists of polygon points."""

    # Polygon parameters.
    # if radii is None:
    # radius_1 = np.random.randint(10, 40)  # np.random.randint(10, 14)
    # radius_2 = radius_1  # if category==1 else np.random.randint(10, 40)
    # else:
    radius_1, radius_2 = radii

    if sides is None:
        possible_sides = random.sample(list(range(3, 8)), 2)
        sides_1 = possible_sides[0]
        sides_2 = possible_sides[1]

    if rotations is None:
        rotation_1 = math.radians(random.randint(0, 360))
        rotation_2 = math.radians(random.randint(0, 360))

    # I need to calculate min_dev_1 based on the actual points not based on the maximum posible enclosing circle...

    if not regular and irregularity is None:
        max_dev_factor = np.random.choice([0.3, 0.4, 0.5, 0.6])
    else:
        max_dev_factor = irregularity
    max_dev_1 = int(radius_1 * max_dev_factor)
    min_dev_1 = radius_1 + max_dev_1
    max_dev_2 = int(radius_2 * max_dev_factor)
    min_dev_2 = radius_2 + max_dev_2

    translation_a = [
        np.random.randint(min_dev_1, canvas_size[0] - min_dev_1),
        np.random.randint(min_dev_1, canvas_size[1] - min_dev_1),
    ]

    translation_b = [
        np.random.randint(min_dev_2, canvas_size[0] - min_dev_2),
        np.random.randint(min_dev_2, canvas_size[1] - min_dev_2),
    ]

    # Generate points.
    if category == 0 and regular:
        # A math.pi/4 (45 degrees) rotation gives the most stable polygons in the "1" category.
        points_a, _ = regular_polygon(
            sides=sides_1,
            radius=radius_1,
            rotation=rotation_1,
            translation=translation_a,
        )
        points_b, _ = regular_polygon(
            sides=sides_2,
            radius=radius_2,
            rotation=rotation_2,
            translation=translation_b,
        )

    elif category == 1 and regular:
        points_a, original_a = regular_polygon(
            sides=sides_1,
            radius=radius_1,
            rotation=rotation_1,
            translation=translation_a,
        )
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in original_a
        ]

    elif category == 0 and not regular:
        points_a, _ = irregular_polygon_from_regular(
            sides=sides_1,
            radius=radius_1,
            rotation=rotation_1,
            translation=translation_a,
            max_dev=max_dev_1,
        )
        points_b, _ = irregular_polygon_from_regular(
            sides=sides_2,
            radius=radius_2,
            rotation=rotation_2,
            translation=translation_b,
            max_dev=max_dev_2,
        )

    elif category == 1 and not regular:
        points_a, original_a = irregular_polygon_from_regular(
            sides=sides_1,
            radius=radius_1,
            rotation=rotation_1,
            translation=translation_a,
            max_dev=max_dev_1,
        )
        points_b = [
            [sum(pair) for pair in zip(point, translation_b)] for point in original_a
        ]

    else:
        raise ValueError("wrong category or regular args!")

    return points_a, points_b, tuple(translation_b), radius_1


# Core graphical functions
def regular_polygon(sides, radius=10, rotation=0, translation=None):
    """Calculates the vertices of a regular polygon by sweeping out a circle, and puting n equally spaced points on it."""
    # The first thing to do is work out the angle (in radians) of each wedge from the center outwards.
    # The total number of radians in a circle is 2 pi, so our value is 2 pi / n per segment.
    one_segment = math.pi * 2 / sides
    # After that a bit of basic trig gives us our points. At this point we scale by our desired radius,
    # and have the opportunity to offset the rotation by a fixed amount too.
    points = [
        (
            int(math.sin(one_segment * i + rotation) * radius),
            int(math.cos(one_segment * i + rotation) * radius),
        )
        for i in range(sides)
    ]

    original_points = copy.copy(points)
    # After that we translate the values by a certain amount, because you probably want your polygon
    # in the center of the screen, not in the corner.
    if translation:
        points = [[sum(pair) for pair in zip(point, translation)] for point in points]
    return points, original_points


# Arrows
def rotate(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin.
    Because in OpenCV the y-axis is inverted this function swaps the x and y axis.
    Args:
        origin: (x, y) tuple.
        point: the point (x, y) to rotate.
        angle: in radiants.

    The angle should be given in radians.
    """
    oy, ox = origin
    py, px = point

    qx = ox + int(math.cos(angle) * (px - ox)) - int(math.sin(angle) * (py - oy))
    qy = oy + int(math.sin(angle) * (px - ox)) + int(math.cos(angle) * (py - oy))
    return int(qy), int(qx)


def rotate_and_translate(origin, point_list, angle, translation):
    """Rotate polygon points counterclockwise by a given angle around a given origin and translate.
    Args:
        origin: (x, y) tuple.
        point_list: list of points (x, y) to rotate.
        angle: in degrees.
    Returns:
        New list of points rotated and translated.
    """
    # Get angle in ratiants.
    radiants = math.radians(angle)

    # Rotate all points.
    new_points = [rotate(origin=origin, point=p, angle=radiants) for p in point_list]

    # Translate all points.
    new_points = [
        [sum(pair) for pair in zip(point, translation)] for point in new_points
    ]

    return new_points


def get_triangle_top_midpoint(point_list):
    """Returns the midpoint of the top of a triangle regardless of the orientation."""

    y = int(min([x[1] for x in point_list]))
    x = int((min([x[0] for x in point_list]) + max([x[0] for x in point_list])) / 2)

    return x, y


def get_triangle_bottom_midpoint(point_list):
    """Returns the midpoint of the top of a triangle regardless of the orientation."""

    y = int(max([x[1] for x in point_list]))
    x = int((min([x[0] for x in point_list]) + max([x[0] for x in point_list])) / 2)

    return x, y


def get_arrow_points(
    radius, center, rotation=0, shape_a="normal", shape_b="normal", continuous=True
):
    """Calculates the points for a arrow.
    Args:
        radius: of the base circle to build the triangles (heads). 5, 7, 9 works well.
        rotation: of the arrow in degrees.
        center: center of the arrow.
        shape_a: shape of head a. "normal", "inverted".
        shape_b: shape of head b. "normal", "inverted".
        continuous: wether the line touches the available heads.
    Returns:
        3 lists of lists of points. the first is the "top" head, the second the "bottom" and the third is the line.
    """

    # The base arrow is based on 4 circles.
    # The overall centre is at 2 radi from the top head centre.
    origin_top = (center[0], int(center[1] - 2 * radius))
    origin_bottom = [center[0], int(center[1] + 2 * radius)]

    points_top, cero_centered_top = regular_polygon(
        sides=3, radius=radius, rotation=math.radians(180), translation=origin_top
    )
    # Use the same function to generate the bottom!
    points_bottom, cero_centered_bottom = regular_polygon(
        sides=3, radius=radius, rotation=math.radians(0), translation=origin_bottom
    )

    # Get line points.
    top_mid_point = get_triangle_bottom_midpoint(points_top)
    bottom_mid_point = get_triangle_top_midpoint(points_bottom)

    # If the arrow isn't continious shrink the line.
    if not continuous:
        separation = int(radius)
        top_mid_point = center[0], top_mid_point[1] + separation
        bottom_mid_point = center[0], bottom_mid_point[1] - separation

    points_line = [top_mid_point, bottom_mid_point]

    if shape_a == "inverted":
        # - radius/2.
        origin_top = [origin_top[0], int(origin_top[1] - radius / 2)]
        points_top, cero_centered_top = regular_polygon(
            sides=3, radius=radius, rotation=math.radians(0), translation=origin_top
        )

    if shape_b == "inverted":
        # + radius/2.
        origin_bottom = [origin_bottom[0], int(origin_bottom[1] + radius / 2) + 1]
        points_bottom, cero_centered_bottom = regular_polygon(
            sides=3,
            radius=radius,
            rotation=math.radians(180),
            translation=origin_bottom,
        )

    # Get angle in ratiants.
    radiants = math.radians(rotation)

    # Rotate all elements the given amount.
    points_top = [rotate(origin=center, point=p, angle=radiants) for p in points_top]
    points_bottom = [
        rotate(origin=center, point=p, angle=radiants) for p in points_bottom
    ]
    points_line = [rotate(origin=center, point=p, angle=radiants) for p in points_line]

    return points_top, points_bottom, points_line


def sample_midpoints_arrows(size, canvas_size):
    xs = random.sample(list(range(size * 4, canvas_size[0] - size * 4)), 2)
    ys = random.sample(list(range(size * 4, canvas_size[1] - size * 4)), 2)
    point_1 = [xs[0], ys[0]]
    point_2 = [xs[1], ys[1]]

    return point_1, point_2


def sample_midpoints_lines(sizes, canvas_size):
    size_1, size_2 = sizes
    x_1 = random.sample(
        list(range(int(size_1 / 2) + 2, canvas_size[0] - int(size_1 / 2 + 2))), 1
    )[0]
    y_1 = random.sample(
        list(range(int(size_1 / 2) + 2, canvas_size[1] - int(size_1 / 2 + 2))), 1
    )[0]
    x_2 = random.sample(
        list(range(int(size_2 / 2) + 2, canvas_size[0] - int(size_2 / 2 + 2))), 1
    )[0]
    y_2 = random.sample(
        list(range(int(size_2 / 2) + 2, canvas_size[1] - int(size_2 / 2 + 2))), 1
    )[0]
    point_1 = (x_1, y_1)
    point_2 = (x_2, y_2)

    return point_1, point_2


def compare_xy(point_1, point_2):
    # Is the lower object to the right of the upper object?
    lower_obj = point_1 if point_1[1] >= point_2[1] else point_2
    upper_obj = point_1 if lower_obj is point_2 else point_2
    comparison = 1 if lower_obj[0] >= upper_obj[0] else 0
    return comparison


# Straight lines
def get_line_points(size, rotation, center):
    radius = size / 2
    angle_1 = math.radians(rotation)
    angle_2 = math.radians(rotation + 180)

    x_1 = int(center[0] + int(radius * math.cos(angle_1)))
    y_1 = int(center[1] + int(radius * math.sin(angle_1)))

    x_2 = int(center[0] + int(radius * math.cos(angle_2)))
    y_2 = int(center[1] + int(radius * math.sin(angle_2)))

    return [(x_1, y_1), (x_2, y_2)]


def open_rectangle(radius=8, x_offset=None, rotation=None, translation=None):
    if rotation is None:
        rotation = 1 * math.pi * np.random.random_sample()

    if x_offset is None:
        x_offset = np.random.randint(8)

    sides = 4
    one_segment = math.pi * 2 / sides
    points = [
        (
            math.sin(one_segment * i + rotation) * radius,
            math.cos(one_segment * i + rotation) * radius,
        )
        for i in range(sides)
    ]

    line_1 = points[0:2]
    line_2 = points[2:4]
    line_2 = [[p[0] - x_offset, p[1]] for p in line_2]
    original_lines = copy.copy([line_1, line_2])

    if translation:
        line_1 = [[sum(pair) for pair in zip(point, translation)] for point in line_1]
        line_2 = [[sum(pair) for pair in zip(point, translation)] for point in line_2]
    lines = [line_1, line_2]

    return lines, original_lines


def ccw_sort(polygon_points):
    """Sort the points counter clockwise around the mean of all points. The sorting can be imagined like a
    radar scanner, points are sorted by their angle to the x axis."""
    polygon_points = np.array(polygon_points)
    mean = np.mean(polygon_points, axis=0)
    d = polygon_points - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return polygon_points[np.argsort(s), :]


def irregular_polygon_from_regular(
    sides, radius=1, rotation=0, translation=None, max_dev=0
):
    # Get regular polygon.
    points, original_points = regular_polygon(
        sides=sides, radius=radius, rotation=rotation, translation=translation
    )

    # Add noise.
    noise = [
        [
            np.random.randint(-max_dev, max_dev + 1),
            np.random.randint(-max_dev, max_dev + 1),
        ]
        for x in points
    ]
    points = [[x[0] + y[0], x[1] + y[0]] for x, y in zip(points, noise)]
    original_points = [
        [x[0] + y[0], x[1] + y[0]] for x, y in zip(original_points, noise)
    ]

    # Return points and cero-centerd points.
    return ccw_sort(points), ccw_sort(original_points)


def divide_polygon(points):
    """Divides polygon at the midsection of every side.
    Args:
        points: list of points.
    Returns:
        List of lits of points."""
    mid_points = []
    for i in range(len(points)):
        if i == len(points) - 1:
            midpoint = [
                (points[i][0] + points[0][0]) / 2,
                (points[i][1] + points[0][1]) / 2,
            ]
        else:
            midpoint = [
                (points[i][0] + points[i + 1][0]) / 2,
                (points[i][1] + points[i + 1][1]) / 2,
            ]
        mid_points.append(midpoint)

    new_points = []
    for i in range(len(mid_points)):
        if i == len(mid_points) - 1:
            new_points.append([mid_points[i], points[i], points[0]])
        else:
            new_points.append([mid_points[i], points[i], points[i + 1]])

    return new_points


def displace_line_around_origin(point_list, d):
    """Displace a line (list of points) away from the center (0, 0) d units."""
    point = point_list[1]
    x, y = point
    d_x = d if x >= 0 else -d
    d_y = d if y >= 0 else -d
    displacement = [d_x, d_y]

    displaced_point_list = [
        [sum(pair) for pair in zip(point, displacement)] for point in point_list
    ]

    return displaced_point_list


def displace_polygon_vertices(list_of_points, radius):
    """Displace polygon subseccions randomly around the center.
    The displacement keeps the angles of the original polygon.
    This function assumes that points are the original polygon
    points around the coordinate (0,0).

    Args:
        points: list of points.
    Returns:
        List of lits of points."""

    mid_points = []
    for i in range(len(list_of_points)):
        if i == len(list_of_points) - 1:
            midpoint = [
                (list_of_points[i][0] + list_of_points[0][0]) / 2,
                (list_of_points[i][1] + list_of_points[0][1]) / 2,
            ]
        else:
            midpoint = [
                (list_of_points[i][0] + list_of_points[i + 1][0]) / 2,
                (list_of_points[i][1] + list_of_points[i + 1][1]) / 2,
            ]
        mid_points.append(midpoint)

    new_points = []
    for i in range(len(mid_points)):
        if i == len(mid_points) - 1:
            new_points.append([mid_points[i], list_of_points[0], mid_points[0]])
        else:
            new_points.append([mid_points[i], list_of_points[i + 1], mid_points[i + 1]])

    # All posible displacements to sample from.
    all_d = list(range(0, radius))
    random.shuffle(all_d)

    # Displace the points from the distance a randomly chosen amount.
    displaced_points = []
    counter = 0
    for point_list in new_points:
        d = all_d[counter]  # random.sample(all_d, 1)[0]
        new_point_list = displace_line_around_origin(point_list, d)
        displaced_points.append(new_point_list)
        counter += 1
        # Reset the counter if reach the end of all displacements.
        if counter >= len(all_d) - 1:
            counter = 0

    return displaced_points


def scramble_poligon(img, midpoint, radius, background):
    # Augment the radius to cover all pixels in teh target patch.
    radius += 1
    # Get start points and end points fo the 4 quadrants.
    sp_1 = (midpoint[0] - radius, midpoint[1] - radius)
    ep_1 = midpoint

    sp_2 = (midpoint[0], midpoint[1] - radius)
    ep_2 = (midpoint[0] + radius, midpoint[1])

    sp_3 = (midpoint[0] - radius, midpoint[1])
    ep_3 = (midpoint[0], midpoint[1] + radius)

    sp_4 = midpoint
    ep_4 = (midpoint[0] + radius, midpoint[1] + radius)

    # Sample offsets.
    if len(range(0, int(radius / 2))) < 4:
        assert (
            False
        ), "Size is too small for scrambling, please increase it at least to 7"
    off_x = random.sample(list(range(0, int(radius / 2))), 4)
    off_y = random.sample(list(range(0, int(radius / 2))), 4)

    # Add offsets.
    new_sp_1 = (sp_1[0] - off_x[0], sp_1[1] - off_y[0])
    new_ep_1 = (ep_1[0] - off_x[0], ep_1[1] - off_y[0])

    new_sp_2 = (sp_2[0] + off_x[1], sp_2[1] - off_y[1])
    new_ep_2 = (ep_2[0] + off_x[1], ep_2[1] - off_y[1])

    new_sp_3 = (sp_3[0] - off_x[2], sp_3[1] + off_y[2])
    new_ep_3 = (ep_3[0] - off_x[2], ep_3[1] + off_y[2])

    new_sp_4 = (sp_4[0] + off_x[3], sp_4[1] + off_y[3])
    new_ep_4 = (ep_4[0] + off_x[3], ep_4[1] + off_y[3])

    # Copy patches.
    patch_1 = np.copy(img[sp_1[1] : ep_1[1], sp_1[0] : ep_1[0]])
    patch_2 = np.copy(img[sp_2[1] : ep_2[1], sp_2[0] : ep_2[0]])
    patch_3 = np.copy(img[sp_3[1] : ep_3[1], sp_3[0] : ep_3[0]])
    patch_4 = np.copy(img[sp_4[1] : ep_4[1], sp_4[0] : ep_4[0]])

    # Wipe out patches in img.
    img[sp_1[1] : ep_1[1], sp_1[0] : ep_1[0]] = background
    img[sp_2[1] : ep_2[1], sp_2[0] : ep_2[0]] = background
    img[sp_3[1] : ep_3[1], sp_3[0] : ep_3[0]] = background
    img[sp_4[1] : ep_4[1], sp_4[0] : ep_4[0]] = background

    # Paste patches in new locations.
    img[new_sp_1[1] : new_ep_1[1], new_sp_1[0] : new_ep_1[0]] = patch_1
    img[new_sp_2[1] : new_ep_2[1], new_sp_2[0] : new_ep_2[0]] = patch_2
    img[new_sp_3[1] : new_ep_3[1], new_sp_3[0] : new_ep_3[0]] = patch_3
    img[new_sp_4[1] : new_ep_4[1], new_sp_4[0] : new_ep_4[0]] = patch_4

    return img
