import numpy as np
from PIL import ImageDraw

from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import apply_antialiasing


class DrawEbbinghaus(DrawStimuli):
    def create_ebbinghaus(
        self,
        r_c,
        d=0,
        r2=0,
        n=0,
        shift=0,
        colour_center_circle=(255, 255, 255),
        img=None,
    ):
        """
        Parameters r_c, d, and r2, are relative to the total image size.
        If you only want to generate the center circle, leave d to 0.
        r_c : radius of the center circle
        d : distance to flankers
        r2 : radius flankers
        n : 0 number flankers
        shift : flankers rotations around center circle [0, pi]
        """
        if img is None:
            img = self.create_canvas()
        draw = ImageDraw.Draw(img)
        if d != 0:
            thetas = np.linspace(0, np.pi * 2, n, endpoint=False) + shift
            dd = img.size[0] * d
            vect = [[np.cos(t) * dd, np.sin(t) * dd] for t in thetas]
            [
                self.circle(draw, np.array(vv) + img.size[0] / 2, img.size[0] * r2 / 2)
                for vv in vect
            ]
        self.circle(
            draw,
            np.array(img.size) / 2,
            img.size[0] * r_c / 2,
            fill=colour_center_circle,
        )

        return apply_antialiasing(img) if self.antialiasing else img

    def create_random_ebbinghaus(
        self,
        r_c,
        n=0,
        flankers_size_range=(0.1, 0.5),
        colour_center_circle=(255, 255, 255),
    ):
        gen_rnd = lambda r: np.random.uniform(*r)

        img = self.create_canvas()
        draw = ImageDraw.Draw(img)

        for i in range(n):
            random_points = [
                np.random.randint(self.canvas_size[0]),
                np.random.randint(self.canvas_size[1]),
            ]
            random_size = self.canvas_size[0] * gen_rnd(
                np.array(flankers_size_range) / 2
            )
            self.circle(draw, np.array(random_points), int(random_size))
        self.circle(
            draw,
            np.array(self.canvas_size) / 2,
            self.canvas_size[0] * r_c / 2,
            fill=colour_center_circle,
        )

        return apply_antialiasing(img) if self.antialiasing else img
