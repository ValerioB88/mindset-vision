import argparse
import csv
import os
import cv2
import pathlib
import PIL.Image as Image
import numpy as np
import sty
import toml
import inspect

from torchvision import transforms
from tqdm import tqdm

from .utils import (
    get_image_facial_landmarks,
    get_bounding_rectangle,
    apply_thatcher_effect_on_image,
)
from src.utils.misc import (
    delete_and_recreate_path,
)

category_folder = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
name_dataset = os.path.basename(os.path.dirname(__file__))


DEFAULTS = {
    "canvas_size": [224, 224],
    "face_folder": "assets/celebA_sample/normal",
    "output_folder": f"data/{category_folder}/{name_dataset}",
    "behaviour_if_present": "overwrite",
}


def generate_all(
    face_folder=DEFAULTS["face_folder"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
    behaviour_if_present=DEFAULTS["behaviour_if_present"],
):
    loc = locals()
    args = {i: loc[i] for i in inspect.getfullargspec(generate_all)[0]}
    config = {f"{category_folder}/{name_dataset}": args}

    face_folder = pathlib.Path(face_folder)
    output_folder = pathlib.Path(output_folder)

    if output_folder.exists() and behaviour_if_present == "skip":
        print(sty.fg.yellow + f"Dataset already exists. Skipping" + sty.rs.fg)
        return str(output_folder)

    delete_and_recreate_path(output_folder)
    toml.dump(config, open(str(output_folder / "config.toml"), "w"))

    [
        (output_folder / cond).mkdir(parents=True, exist_ok=True)
        for cond in [
            "straight",
            "inverted",
            "thatcherized_straight",
            "thatcherized_inverted",
        ]
    ]
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("assets/lbfmodel.yaml")
    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Transformation", "FaceId"])
        for idx, image_path in tqdm(enumerate(face_folder.glob("*"))):
            image_facial_landmarks = get_image_facial_landmarks(image_path, facemark)
            image_name = image_path.stem
            if (
                not image_facial_landmarks
                or len(image_facial_landmarks) == 0
                or len(image_facial_landmarks) != 68
            ):
                continue
            left_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[36:42])
            right_eye_rectangle = get_bounding_rectangle(image_facial_landmarks[42:48])
            mouth_rectangle = get_bounding_rectangle(image_facial_landmarks[48:68])
            cv_image = apply_thatcher_effect_on_image(
                str(image_path),
                np.array(left_eye_rectangle).astype(int),
                np.array(right_eye_rectangle).astype(int),
                np.array(mouth_rectangle).astype(int),
            )
            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.fromarray(cv_image)
            ).save(output_folder / "thatcherized_straight" / f"{idx}.png")
            writer.writerow(
                [
                    pathlib.Path("thatcherized_straight") / f"{image_name}.png",
                    "thatcherized_straight",
                    idx,
                ]
            )

            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.fromarray(cv2.flip(cv_image, 0))
            ).save(output_folder / "thatcherized_inverted" / f"{idx}.png")
            writer.writerow(
                [
                    pathlib.Path("thatcherized_inverted") / f"{image_name}.png",
                    "thatcherized_inverted",
                    idx,
                ]
            )
            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.open(image_path)
            ).save(output_folder / "straight" / f"{image_name}.png")
            writer.writerow([pathlib.Path("straight") / f"{idx}.png", "straight", idx])

            transforms.CenterCrop((canvas_size[1], canvas_size[0]))(
                Image.open(image_path).rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            ).save(output_folder / "inverted" / f"{image_name}.png")
            writer.writerow(
                [pathlib.Path("inverted") / f"{image_name}.png", "inverted", idx]
            )
    return str(output_folder)


if __name__ == "__main__":
    description = "We provide a small face celebrity dataset using a subset of CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, but the user can specify any folder containing images of faces. Each image is resized according to parameters specified by the user and then reoriented into both an upright and a 180-degree inverted configuration. Furthermore, it is either 'Thatcherized' or unaltered. To `Thatcherize' an image we compute the landmarks of the eyes and the mouth, compute the bounding rectangle for each, and rotate them around their centre of mass. Blurring on the edge is applied to minimize artefacts."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        default=DEFAULTS["output_folder"],
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure used to create the dataset",
    )

    parser.add_argument(
        "--face_folder",
        "-ff",
        default=DEFAULTS["face_folder"],
        help="The folder containing faces that need to be Thatcherized. These faces will also be resized to `canvas_size` size. ",
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default=DEFAULTS["canvas_size"],
        help="The size of the canvas. If called through command line, a string in the format NxM.",
        type=lambda x: [int(i) for i in x.split("x")],
    )
    parser.add_argument(
        "--behaviour_if_present",
        "-if_pres",
        help="What to do if the dataset folder is already present? Choose between [overwrite], [skip]",
        dest="behaviour_if_present",
        default=DEFAULTS["behaviour_if_present"],
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
