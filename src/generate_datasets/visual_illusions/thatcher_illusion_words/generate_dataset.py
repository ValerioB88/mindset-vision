"""
This script is used to create the train/test data
"""

import argparse
import csv
import inspect
import json
import math
import os
import pathlib
import random
from itertools import product
from pathlib import Path
import uuid
from bs4 import Stylesheet
import sty
from numpy import arange
import toml
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm
from src.generate_datasets.visual_illusions.thatcher_illusion_words.utils import (
    CreateData,
    read_corpus,
)
from src.utils.drawing_utils import DrawStimuli
from src.utils.misc import DEFAULTS as BASE_DEFAULTS, delete_and_recreate_path

DEFAULTS = BASE_DEFAULTS.copy()

from src.utils.misc import add_general_args

category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))

DEFAULTS.update(
    {
        "jittery": 0.04,
        "num_words": 100,
        "num_samples_per_word": 5,
        "num_letters_per_word": [5, 9],
        "num_letters_to_rotate": 2,
        "size_fonts": [18, 35],
        "use_random_words": False,
        "output_folder": f"data/{category_folder}/{name_dataset}",
    }
)

# exclude letters that are either invariant to 180 rotations or that they will look like other letter after 180 rotations
letters_not_to_rotate = ["O", "W", "M", "N"]


def generate_all(
    jittery=DEFAULTS["jittery"],
    num_words=DEFAULTS["num_words"],
    num_samples_per_word=DEFAULTS["num_samples_per_word"],
    num_letters_per_word=DEFAULTS["num_letters_per_word"],
    size_fonts=DEFAULTS["size_fonts"],
    num_letters_to_rotate=DEFAULTS["num_letters_to_rotate"],
    use_random_words=DEFAULTS["use_random_words"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    background_color=DEFAULTS["background_color"],
    antialiasing=DEFAULTS["antialiasing"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    """Create train data at the data folder.
    Args:
        dummy (bool, optional): make smaller dataset.
        Defaults to False.
        random (bool, optional): use random strings.
        Defaults to False.
    """
    word_folder = Path("assets", "words")
    font_folder = word_folder.joinpath("fonts")
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(Stylesheet.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))
    conditions = [
        "straight",
        "inverted",
        "thatcherized_straight",
        "thatcherized_inverted",
    ]
    [(output_folder / cond).mkdir(parents=True, exist_ok=True) for cond in conditions]

    if use_random_words:
        corpus = json.load(open(Path(word_folder, "random_strings.json"), "r"))
    else:
        corpus: list[str] = read_corpus(Path(word_folder, "1000-corpus.txt"))

    if isinstance(num_letters_per_word, list):
        corpus = [
            i
            for i in corpus
            if num_letters_per_word[0] <= len(i) <= num_letters_per_word[1]
        ]
    else:
        corpus = [i for i in corpus if len(i) == num_letters_per_word]

    fonts: list = [f for f in os.listdir(font_folder) if f.endswith(".ttf")]
    w = corpus[0]
    # with this we get rid of all the words which, when excluded their unrotable letters, they don't have enough roteable letters.
    corpus = [
        w
        for w in corpus
        if len([ll for ll in w if ll not in letters_not_to_rotate])
        >= (
            min(num_letters_to_rotate)
            if isinstance(num_letters_to_rotate, list)
            else num_letters_to_rotate
        )
    ]
    corpus = random.sample(corpus, min(len(corpus), num_words))
    create = CreateData(
        canvas_size=canvas_size,
        background=background_color,
        antialiasing=antialiasing,
        variance_font=0,
        coefficient_translation=jittery,
        coefficient_space=1.5,
        word_folder=word_folder,
        font_folder=font_folder,
    )
    # corpus = ["mindset"]
    # size_fonts = 35
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "Condition",
                "Word",
                "LettersRotated",
                "IdxLetterRotated",
                "NameFont",
                "SizeFont",
                "IterNum",
            ]
        )
        for w in tqdm(corpus):
            w = w.upper()
            for count in tqdm(range(num_samples_per_word), leave=False):
                name_font = random.sample(fonts, 1)[0]
                size_font = (
                    random.randint(size_fonts[0], size_fonts[1])
                    if isinstance(size_fonts, list)
                    else size_fonts
                )

                nl = (
                    random.sample(
                        range(num_letters_to_rotate[0], num_letters_to_rotate[1] + 1),
                        1,
                    )[0]
                    if isinstance(num_letters_to_rotate, list)
                    else num_letters_to_rotate
                )
                nl = min(len(w), nl)
                idx_letter_rotateable = [
                    idx for idx, ww in enumerate(w) if ww not in letters_not_to_rotate
                ]
                idx_letters_to_rotate = random.sample(
                    idx_letter_rotateable, min(len(idx_letter_rotateable), nl)
                )
                for cond in conditions:
                    ltr = [] if "thatcherized" not in cond else idx_letters_to_rotate
                    canvas = create.create_images(w, name_font, size_font, ltr)
                    if "inverted" in cond:
                        canvas = canvas.rotate(180)
                    uui = str(uuid.uuid4().hex[:8])
                    img_path = f"{cond}/{w}_{count}_{uui}.png"
                    canvas.save(output_folder / img_path)
                    writer.writerow(
                        [
                            img_path,
                            cond,
                            w,
                            [w[i] for i in idx_letters_to_rotate],
                            idx_letters_to_rotate,
                            name_font,
                            size_font,
                            count,
                        ]
                    )


if __name__ == "__main__":
    description = "We employ a collection of 1000 natural English words or artificially generated sequences of random letters. All entries are uniformly presented in uppercase, covering a range from 3 to 8 letters in length.  Words are always presented uppercase. The corpus contains words ranging from 3 to 8 letters. Following Wong et al. (2010), to simulate the Thatcher Effect for words, we rotate 180 degree one or more letters. To increase variability, each word is displayed in one of ten different fonts, with variable font sizes, and includes jitter on each letter. The configurable parameters include the number of words, the exact or range of letter counts per word, the number or range of letters to be rotated, the font size range, the level of jitter, and whether to use random strings or natural English words.\nREF: Wong, Yetta K, Elyssa Twedt, David Sheinberg, and Isabel Gauthier. `Does Thompson's Thatcher Effect Reflect a Face-Specific Mechanism?` Perception 39, no. 8 (1 August 2010): 1125-41. https://doi.org/10.1068/p6659."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_general_args(parser)
    parser.set_defaults(output_folder=DEFAULTS["output_folder"])
    parser.add_argument(
        "--num_words",
        "-nw",
        default=DEFAULTS["num_words"],
        help="Number of unique words to use. Each will be used `num_samples_per_word` times. Max is 1000 words.",
        type=int,
    )
    parser.add_argument(
        "--jittery",
        "-j",
        default=DEFAULTS["jittery"],
        help="The amount of translational jittery for each letter",
        type=float,
    )

    parser.add_argument(
        "--size_fonts",
        "-sf",
        default=DEFAULTS["size_fonts"],
        help="The size(s) to use for the fonts. Could be a number or a range. From commad line, use the format MIN_MAX (inclusive) for ranges",
        type=lambda x: [int(i) for i in x.split("_")] if "_" in x else x,
    )
    parser.add_argument(
        "--num_samples_per_word",
        "-ns",
        default=DEFAULTS["num_samples_per_word"],
        help="",
        type=int,
    )
    parser.add_argument(
        "--num_letters_per_word",
        "-nlw",
        default=DEFAULTS["num_letters_per_word"],
        help="Number of letters per word. The corpus used contains words from 3 to 7 letters. Either a range or a unique number. From command line, use the format MIN_MAX (inclusive) for ranges",
        type=lambda x: [int(i) for i in x.split("_")] if "_" in x else x,
    )
    parser.add_argument(
        "--num_letters_to_rotate",
        "-nlr",
        default=DEFAULTS["num_letters_to_rotate"],
        help="Number of letters to rotate for each word, capped at each word's length.  From command line, use the format MIN_MAX (inclusive) for ranges",
        type=lambda x: [int(i) for i in x.split("_")] if "_" in x else x,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
