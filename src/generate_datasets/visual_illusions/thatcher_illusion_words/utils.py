import math
import os
import random
from itertools import product
from operator import mul
from pathlib import Path
from bs4 import Stylesheet
import numpy as np
from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
from scipy import ndimage
from PIL import ImageDraw
import toml
from tqdm import tqdm
from src.utils.drawing_utils import DrawStimuli


def read_corpus(path: Path):
    corpus = open(path, "r").read()
    corpus: list[str] = [w for w in corpus.split("\n") if w != ""]
    corpus: list[str] = [w for w in corpus if len(w)]
    return corpus


class CreateData(DrawStimuli):
    def __init__(
        self,
        variance_font,
        coefficient_space,
        coefficient_translation,
        word_folder,
        font_folder,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.font_folder = font_folder
        self.word_folder = word_folder
        self.variance_font = variance_font
        self.coefficient_space = coefficient_space
        self.coefficient_translate = coefficient_translation

    def find_letter_bbox(self, image):
        """
        Find the bounding box of the letter in the image based on the letter's color.

        Args:
        - image: PIL.Image object with the letter drawn on it.
        - letter_color: The color of the letter as a tuple (R, G, B).

        Returns:
        - A 4-tuple (left, upper, right, lower) defining the bounding box of the letter.
        """
        pixels = image.load()
        width, height = image.size

        # Initialize min/max values with opposite extremes
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        # Scan all pixels to find the letter based on its color
        for x in range(width):
            for y in range(height):
                if not pixels[x, y][:3] == self.background:
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    max_x, max_y = max(max_x, x), max(max_y, y)

        # Return the bounding box (left, upper, right, lower)
        return min_x, min_y, max_x + 1, max_y + 1

    def textsize_for_drawing(self, text, font):
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def real_bbox_text(self, text, font):
        shape_letter = self.textsize_for_drawing(text, font=font)
        canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0))
        Draw(canvas_letter).text((0, 0), text, fill=self.fill, font=font)
        bbox = self.find_letter_bbox(canvas_letter)
        return bbox

    def create_images(self, word, name_font, size_font, idx_letters_to_rotate):
        self.word = word.upper()
        self.name_font = name_font
        self.size_font = size_font

        canvas = self.create_canvas()
        self.width, self.height = canvas.size[0], canvas.size[1]

        self.center: tuple = (self.width / 2, self.height / 2)

        self.letters_size_font_shift: list = [
            random.randint(-self.variance_font, self.variance_font)
            for _ in range(0, len(self.word))
        ]
        self.width_letters_cumulative = self.get_width_letters(Draw(canvas))
        self.shape_half_word: tuple = (
            self.width_letters_cumulative[-1] / 2,
            self.get_max_height(Draw(canvas)) / 2,
        )
        self.average_diagonal_length = self.get_average_diagonal_length(Draw(canvas))
        self.radius_translate = self.get_radius_translate()

        self.initial_h_pos = (
            self.width * 0.5 + (0.03 * self.width) - self.shape_half_word[0]
        )
        self.v_pos_base = self.height * 0.5  # + self.shape_half_word[1]
        for i in range(0, len(self.word)):
            letter_font = self.get_zoomed_font(self.letters_size_font_shift[i])

            shape_letter = self.textsize_for_drawing(self.word[i], font=letter_font)
            canvas_letter = new("RGBA", shape_letter, color=(0, 0, 0))
            Draw(canvas_letter).text(
                (0, 0), self.word[i], fill=self.fill, font=letter_font
            )
            bbox = self.real_bbox_text(self.word[i], letter_font)
            width_l, height_l = bbox[2] - bbox[0], bbox[3] - bbox[1]
            canvas_letter = canvas_letter.crop(bbox)
            if i in idx_letters_to_rotate:
                canvas_letter = self.rotate(
                    canvas_letter,
                    angle=180,
                )

            canvas_letter.putdata(self.set_background_transparent(canvas_letter))
            canvas.paste(
                im=canvas_letter,
                box=self.get_final_position_letter(i, height_l),
                mask=canvas_letter,
            )

        return canvas

    def rotate(self, image, angle: int):
        image_array = np.array(image)
        rotated_array = ndimage.rotate(
            image_array, angle, cval=0.0, reshape=True, mode="constant", prefilter=True
        )
        return Image.fromarray(rotated_array)

    def get_radius_translate(self):
        return self.coefficient_translate * self.average_diagonal_length

    def size_for_drawing(self, text, font):
        im = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def get_w_h_letters(self, text, font):
        bbox = self.real_bbox_text(text, font=font)

        width_l, height_l = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return width_l, height_l

    def get_width_letters(self, draw: Draw) -> list:
        shapes_letters: list = [
            self.size_for_drawing(
                self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i])
            )
            for i in range(0, len(self.word))
        ]
        width_letters: list = [i[0] * self.coefficient_space for i in shapes_letters]
        width_letters.insert(0, 0)
        return np.cumsum(width_letters)

    def get_average_diagonal_length(self, draw: Draw) -> float:
        shapes_letters: list = [
            self.get_w_h_letters(
                self.word[i], font=self.get_zoomed_font(self.letters_size_font_shift[i])
            )
            for i in range(0, len(self.word))
        ]
        return sum([math.sqrt(s[0] ** 2 + s[1] ** 2) for s in shapes_letters]) / len(
            shapes_letters
        )

    def get_max_height(self, draw: Draw) -> float:
        shapes_letters: list = [
            self.get_w_h_letters(
                self.word[i],
                font=truetype(
                    os.path.abspath(Path("assets", "words", "fonts") / self.name_font),
                    self.size_font + self.letters_size_font_shift[i],
                ),
            )
            for i in range(0, len(self.word))
        ]
        return max([s[1] for s in shapes_letters])

    def get_zoomed_font(self, zoom: int):
        return truetype(
            os.path.abspath(Path("assets", "words", "fonts") / self.name_font),
            self.size_font + zoom,
        )

    def set_background_transparent(self, image) -> list:
        return [
            (lambda i: (*self.background, 0) if i[:3] == self.background else i)(i)
            for i in image.getdata()
        ]

    def get_translation_vector(self, radius: float) -> tuple:
        r = radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        return (r * math.cos(theta), r * math.sin(theta))

    def get_final_position_letter(self, instance, h_l) -> tuple:
        position_letter: tuple = (
            self.initial_h_pos + self.width_letters_cumulative[instance],
            self.height * 0.5 - h_l // 2,
        )
        position_letter: tuple = tuple(
            map(
                sum,
                zip(
                    self.get_translation_vector(self.radius_translate), position_letter
                ),
            )
        )
        return (int(position_letter[0]), int(position_letter[1]))
