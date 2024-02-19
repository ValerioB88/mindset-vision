import matplotlib.pyplot as plt


def get_rgba_color(num):
    """
    Returns a unique RGBA color tuple for a given number less than 10.
    """
    assert num >= 0 and num < 10, "Number should be between 0 and 9"
    colors = plt.cm.get_cmap("tab10", 10)  # get a color map with 10 colors
    rgba = colors(num)  # get the RGBA color tuple for the given number
    rgba = tuple([int(255 * x) for x in rgba])
    return rgba
