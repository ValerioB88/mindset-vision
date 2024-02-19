"""
** Note that this demo has to be ran from the root directory of the project.

A demo script to showcase the functionality of the ShapeCoreFunctions class
and the ParentStimuli class in a canvas. The script demonstrates various operations
that can be performed on shapes, such as cutting, rotating, moving, and changing colors.
Additional operations can be added by extending the ShapeCoreFunctions or Shapes classes.
"""

from src.utils.shape_based_image_generation.modules.parent import ParentStimuli
from src.utils.shape_based_image_generation.modules.shapes import Shapes
from src.utils.shape_based_image_generation.utils.parallel import parallel

# The `ShapeCoreFunctions` class in the provided code defines a set of methods for working with shapes on a canvas. It is capable of performing various operations like:

# 1. Cutting the shape along a line defined by a reference point and an angle (`cut` method)
# 2. Shrinking the image to the specified factor (`shrink` method)
# 3. Rotating the image by a specified angle (`rotate` method)
# 4. Moving the shape to specified coordinates (`move_to` method)
# 5. Moving the shape next to another shape in a specified direction (`move_next_to` method)
# 6. Moving the shape towards a direction by a specified distance (`move_towards` method)
# 7. Moving the shape apart from another shape by a specified distance (`move_apart_from` method)
# 8. Getting the distance between the center of the bounding box of the shape and another shape (`get_distance_from` method)
# 9. Checking if the shape overlaps with another shape (`is_touching` method)
# 10. Filling all the non-transparent pixels with the current color (`set_color` method)

# for adding additional operations, you can add methods to the `ShapeCoreFunctions` class. You can also add methods to the `Shapes` class for shapes that are not defined in the `ShapeCoreFunctions` class.


def main(*args):
    canvas = ParentStimuli(target_image_size=224)

    shape_1 = Shapes(parent=canvas)
    shape_2 = Shapes(parent=canvas)

    # add an arc or any custom shapes that you define by
    #   inheriting from the `ShapeCoreFunctions` class
    shape_1.add_arc(size=0.1)
    shape_2.add_arc(size=0.1)

    # cut the shape along a line defined by a reference point and an angle
    piece_1, piece_2 = shape_1.cut((0.5, 0.5), 0)

    # now this is the interesting part
    #   chain the operations together on piece 1
    piece_1.move_next_to(shape_2, "RIGHT").rotate(90).set_color("red")

    # register means attach the shape to the final canvas
    # only registered shapes that you want to keep
    piece_1.register()
    shape_2.register()

    # finalize all the optional operations, e.g. shrink, center, etc.
    # if you want to amend / add more operations, you can do so in the parent class
    canvas.binary_filter().add_background().shrink()
    canvas.canvas.show()  # or canvas.canvas.save("test.png")


if __name__ == "__main__":
    # if you want to make one image, you can just call main()
    main()

    # if you want to make multiple images, you can use the parallel function
    # note that this will by default use all the cores on your machine
    # parallel(make_one=main, n=100)
