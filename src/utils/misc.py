import pathlib
import shutil
from typing import List

import PIL
import numpy as np
import sty
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import tqdm

from src.utils.similarity_judgment.misc import draw_random_from_ranges

try:
    import neptune
    from neptune.types import File

except:
    pass


def assert_exists(path):
    if not os.path.exists(path):
        assert False, sty.fg.red + f"Path {path} doesn't exist!" + sty.rs.fg


def conditional_tqdm(iterable, enable_tqdm, **kwargs):
    if enable_tqdm:
        return tqdm.tqdm(iterable, **kwargs)
    else:
        return iterable


class ConfigSimple:
    def __init__(self, **kwargs):
        self.verbose = True
        [self.__setattr__(k, v) for k, v in kwargs.items()]

    def __setattr__(self, *args, **kwargs):
        super().__setattr__(*args, **kwargs)


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)

    canvas_size = np.shape(image)

    font_scale = np.ceil(canvas_size[1]) / 150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[0, 0, 0],
        lineType=cv2.LINE_AA,
        thickness=6,
    )
    umat = cv2.putText(
        img=cv2.UMat(umat),
        text=text,
        org=(0, int(canvas_size[1] - 3)),
        fontFace=font,
        fontScale=font_scale,
        color=[255, 255, 255],
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    image = cv2.UMat.get(umat)
    image = np.array(image, np.uint8)
    return image


def weblog_dataset_info(
    dataloader,
    log_text="",
    dataset_name=None,
    weblogger=1,
    plotter=None,
    num_batches_to_log=2,
):
    stats = {}

    def simple_plotter(idx, data):
        images, labels, *more = data
        plot_images = images[0 : np.max((4, len(images)))]
        metric_str = "Debug/{} example images".format(log_text)
        lab = [f"{i.item():.3f}" for i in labels]
        if isinstance(weblogger, neptune.Run):
            [
                weblogger[metric_str].log(
                    File.as_image(
                        convert_normalized_tensor_to_plottable_array(
                            im, stats["mean"], stats["std"], text=lb
                        )
                        / 255
                    )
                )
                for im, lb in zip(plot_images, lab)
            ]

    if plotter is None:
        plotter = simple_plotter
    if "stats" in dir(dataloader.dataset):
        dataset = dataloader.dataset
        dataset_name = dataset.name
        stats = dataloader.dataset.stats
    else:
        dataset_name = "no_name" if dataset_name is None else dataset_name
        stats["mean"] = [0.5, 0.5, 0.5]
        stats["std"] = [0.2, 0.2, 0.2]
        Warning(
            "MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch"
        )

    if isinstance(weblogger, neptune.Run):
        weblogger["Logs"] = f'{dataset_name} mean: {stats["mean"]}, std: {stats["std"]}'

    for idx, data in enumerate(dataloader):
        plotter(idx, data)
        if idx + 1 >= num_batches_to_log:
            break

    # weblogger[weblogger_text].log(File.as_image(image))


def imshow_batch(inp, stats=None, labels=None, title_more="", maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats["mean"]
        std = stats["std"]
    """Imshow for Tensor."""

    cols = int(np.ceil(np.sqrt(len(inp))))
    if ax is None:
        fig, ax = plt.subplots(cols, cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis("off")
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            text = (
                str(labels[idx]) + " " + (title_more[idx] if title_more != "" else "")
            )
            # ax[idx].set_title(text, size=5)
            ax[idx].text(
                0.5,
                0.1,
                f"{labels[idx]:.3f}",
                horizontalalignment="center",
                transform=ax[idx].transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )

    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0.01, left=0, right=1, hspace=0.2, wspace=0.01)
    return ax


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def convert_lists_to_strings(obj):
    if isinstance(obj, list):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_strings(v) for k, v in obj.items()}
    else:
        return obj


DEFAULTS = {
    "canvas_size": [224, 224],
    "background_color": [0, 0, 0],
    "antialiasing": True,
    "behaviour_if_present": "overwrite",
}


def add_general_args(parser):
    parser.add_argument(
        "--output_folder",
        "-o",
        help="The folder containing the data. It will be created if doesn't exist. The default will match the folder structure of the generation script",
    )
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default=DEFAULTS["canvas_size"],
        help="The size of the canvas. If called through command line, a string in the format NxM eg `224x224`.",
        type=lambda x: [int(i) for i in x.split("x")] if isinstance(x, str) else x,
    )

    parser.add_argument(
        "--background_color",
        "-bg",
        default=DEFAULTS["background_color"],
        help="Specify the background color. Could be a list of RGB values, or `rnd-uniform` for a random (but uniform) color. If called from command line, the RGB value must be a string in the form R_G_B",
        type=lambda x: (
            [int(i) for i in x.split("_")]
            if "_" in x
            else x if isinstance(x, str) else x
        ),
    )

    parser.add_argument(
        "--antialiasing",
        "-antial",
        dest="antialiasing",
        help="Specify whether we want to enable antialiasing",
        action="store_true",
        default=DEFAULTS["antialiasing"],
    )

    parser.add_argument(
        "--behaviour_if_present",
        "-if_pres",
        help="What to do if the dataset folder is already present? Choose between [overwrite], [skip]",
        default=DEFAULTS["behaviour_if_present"],
    )


def pretty_print_dict(dictionary, indent=0, name=None):
    key_color = sty.fg.blue
    value_color = sty.fg.green
    name_color = sty.fg.red
    reset_color = sty.rs.fg

    if name is not None:
        print(name_color + f"~~~ {name} ~~~" + reset_color)

    for key, value in sorted(dictionary.items()):
        print(" " * indent + key_color + key + reset_color, end=": ")
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 4)
        else:
            print(value_color + str(value) + reset_color)


def update_dict(dictA, dictB, replace=True):
    key_color = sty.fg.blue
    old_value_color = sty.fg.red
    new_value_color = sty.fg.green
    reset_color = sty.rs.fg

    for key in dictB:
        if dictB[key] is not None:
            if (
                key in dictA
                and isinstance(dictA[key], dict)
                and isinstance(dictB[key], dict)
            ):
                # if the value in dictA is a dict and the value in dictB is a dict, recursively update the nested dict
                update_dict(dictA[key], dictB[key], replace)
            else:
                # otherwise, simply update the value in dictA with the value in dictB
                if replace or (not replace and key not in dictA):
                    old_value = dictA[key] if key in dictA else "none"
                    dictA[key] = dictB[key]
                    (
                        print(
                            key_color
                            + f"Updated {key} : "
                            + reset_color
                            + old_value_color
                            + f"{old_value} => "
                            + reset_color
                            + key_color
                            + f"{key}: "
                            + reset_color
                            + new_value_color
                            + f"{dictB[key]}"
                            + reset_color
                        )
                        if old_value != dictB[key]
                        else None
                    )
                else:
                    print(
                        key_color
                        + f"Value {key} "
                        + reset_color
                        + "not replaced as already present ("
                        + old_value_color
                        + f"{dictA[key]}"
                        + reset_color
                        + ") and 'replace=False'"
                    )
    return dictA


def apply_antialiasing(img: PIL.Image, amount=None):
    if amount is None:
        amount = min(img.size) * 0.00334
    return img.filter(ImageFilter.GaussianBlur(radius=amount))


def delete_and_recreate_path(path: pathlib.Path):
    shutil.rmtree(path) if path.exists() else None
    path.mkdir(parents=True, exist_ok=True)
    print(sty.fg.yellow + f"Recreating {path}." + sty.rs.fg)


import random


def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


import os
import subprocess


def check_download_ETH_80_dataset(destination_dir):
    repo_url = "https://github.com/chenchkx/ETH-80/"

    if not os.path.exists(destination_dir):
        print(
            f"ETH-80 dataset, used for the viewpoint invariance dataset, is not found in {destination_dir}. It will be downloaded (~308MB)."
        )
        subprocess.run(["git", "clone", repo_url, destination_dir])
        reorganize_ETH_80(destination_dir)

    else:
        print(f"ETH-80 dataset found in {destination_dir}")


def get_affine_rnd_fun(transf_values):
    transf_ranges = {
        k: [v] if not isinstance(v[0], list) else v for k, v in transf_values.items()
    }

    tr = lambda: [
        (
            draw_random_from_ranges(transf_ranges["translation_X"])
            if "translation_X" in transf_values and transf_values["translation_X"]
            else 0
        ),
        (
            draw_random_from_ranges(transf_ranges["translation_Y"])
            if "translation_Y" in transf_values and transf_values["translation_Y"]
            else 0
        ),
    ]

    scale = (
        (lambda: draw_random_from_ranges(transf_ranges["scale"]))
        if "scale" in transf_values and transf_values["scale"]
        else lambda: 1.0
    )
    rot = (
        (lambda: draw_random_from_ranges(transf_ranges["rotation"]))
        if "rotation" in transf_values and transf_values["rotation"]
        else lambda: 0
    )
    return lambda: {"rt": rot(), "tr": tr(), "sc": scale(), "sh": 0.0}


def my_affine(img, translate, **kwargs):
    return F.affine(
        img,
        translate=[int(translate[0] * img.size[0]), int(translate[1] * img.size[1])],
        **kwargs,
    )


def modify_toml(
    toml_lines,
    modified_key_starts_with="num_samples",
    modify_value_fun=None,
):
    lines = []
    current_header = None  # To store the current header
    for line in toml_lines:
        stripped_line = line.strip()

        # Check if the line is a header
        if stripped_line.startswith("[") and stripped_line.endswith("]"):
            current_header = stripped_line.strip("[]")

        # Check if the line starts with the specified key
        elif stripped_line.startswith(modified_key_starts_with):
            parts = line.split("=")
            key = parts[0].strip()
            value = parts[1].strip()

            # Pass the key, value, and the current header to the modify function
            modified_value = modify_value_fun(current_header, value)
            line = f"{key} = {modified_value}\n"
        lines.append(line)
    return lines


def reorganize_ETH_80(base_dir):
    base_dir = "assets/ETH_80"

    images_dir = os.path.join(base_dir, "images")

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder != "images":
            shutil.move(folder_path, images_dir)

    source_dir = os.path.join(base_dir, "images")
    target_dir = os.path.join(base_dir, "maps")

    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over each class folder
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            # Iterate over each object folder within the class folder
            for object_folder in os.listdir(class_path):
                object_path = os.path.join(class_path, object_folder)
                if os.path.isdir(object_path):
                    # Check if the "map" folder exists in the object folder
                    map_folder_path = os.path.join(object_path, "maps")
                    if os.path.exists(map_folder_path):
                        # Define the target path for the map folder
                        target_map_folder_path = os.path.join(
                            target_dir, class_folder, object_folder
                        )
                        # Create target map folder if it doesn't exist
                        os.makedirs(target_map_folder_path, exist_ok=True)
                        # Move the map folder to the target location
                        for map_file in os.listdir(map_folder_path):
                            shutil.move(
                                os.path.join(map_folder_path, map_file),
                                target_map_folder_path,
                            )
                        shutil.rmtree(map_folder_path)

    print("Map folders have been reorganized.")
