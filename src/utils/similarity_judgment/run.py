import argparse

import toml
import inspect

import torch
from src.utils.device_utils import set_global_device, to_global_device
from src.utils.net_utils import GrabNet, load_pretraining
from src.utils.misc import delete_and_recreate_path, update_dict, pretty_print_dict
from sty import fg, rs
import pickle
import os
import pathlib
import torchvision.transforms as transforms
import torchvision
from src.utils.similarity_judgment.misc import (
    has_subfolders,
    PasteOnCanvas,
)
from src.utils.similarity_judgment.activation_recorder import (
    RecordDistance,
)
import inspect


def compute_distance(
    basic_info=None,
    options=None,
    saving_folders=None,
    transformation=None,
):
    with open(os.path.dirname(__file__) + "/default_distance_config.toml", "r") as f:
        toml_config = toml.load(f)
        toml_config["transformation"]["fill_color"] = tuple(
            toml_config["transformation"]["fill_color"]
        )
    # update the toml_config file based on the input args to this function
    local_vars = locals()
    if not all(transformation["values"].values()):
        transformation["repetitions"] = 1

    update_dict(
        toml_config,
        {i: local_vars[i] for i in inspect.getfullargspec(compute_distance)[0]},
    )

    network, norm_values, resize_value = GrabNet.get_net(
        toml_config["options"]["architecture_name"],
        imagenet_pt=True if toml_config["options"]["imagenet_pretrained"] else False,
    )
    set_global_device(toml_config["options"]["gpu_idx"])

    pathlib.Path(toml_config["saving_folders"]["results_folder"]).mkdir(
        parents=True, exist_ok=True
    )

    pretty_print_dict(toml_config)

    load_pretraining(
        net=network,
        optimizers=None,
        net_state_dict_path=toml_config["network"]["state_dict_path"],
        optimizers_path=None,
    )
    to_global_device(network)
    transf_list = [
        x
        for x in [
            (
                PasteOnCanvas(
                    toml_config["transformation"]["canvas_to_image_ratio"],
                    toml_config["transformation"]["fill_color"],
                )
                if toml_config["transformation"]["copy_on_bigger_canvas"]
                else None
            ),
            transforms.Resize(resize_value),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_values["mean"], norm_values["std"]),
        ]
        if x is not None
    ]

    transform = torchvision.transforms.Compose(transf_list)

    # delete_and_recreate_path(
    #     pathlib.Path(toml_config["saving_folders"]["results_folder"])
    # )

    toml.dump(
        {
            **toml_config,
            "transformation": {
                **toml_config["transformation"],
                "fill_color": list(toml_config["transformation"]["fill_color"]),
            },
            "basic_info": {
                **toml_config["basic_info"],
                "annotation_file_path": str(
                    toml_config["basic_info"]["annotation_file_path"]
                ),
            },
        },
        open(toml_config["saving_folders"]["results_folder"] + "/config.toml", "w"),
    )

    debug_image_path = toml_config["saving_folders"]["results_folder"] + "/debug_img/"
    save_debug_images = toml_config["saving_folders"]["save_debug_images"]

    if save_debug_images:
        pathlib.Path(os.path.dirname(debug_image_path)).mkdir(
            parents=True, exist_ok=True
        )

    recorder = RecordDistance(
        annotation_filepath=toml_config["basic_info"]["annotation_file_path"],
        match_factors=toml_config["basic_info"]["match_factors"],
        non_match_factors=toml_config["basic_info"]["non_match_factors"],
        factor_variable=toml_config["basic_info"]["factor_variable"],
        reference_level=toml_config["basic_info"]["reference_level"],
        filter_factor_level=toml_config["basic_info"]["filter_factor_level"],
        distance_metric=toml_config["options"]["distance_metric"],
        net=network,
        only_save=toml_config["options"]["save_layers"],
    )

    distance_df, layers_names = recorder.compute_from_annotation(
        transform=transform,
        matching_transform=toml_config["transformation"]["matching_transform"],
        fill_bk=toml_config["transformation"]["fill_color"],
        transf_boundaries=toml_config["transformation"]["values"],
        transformed_repetition=toml_config["transformation"]["repetitions"],
        path_save_fig=debug_image_path if save_debug_images else None,
        add_columns=toml_config["basic_info"]["add_columns"],
    )
    save_folder = pathlib.Path(toml_config["saving_folders"]["results_folder"])
    dataframe_path = save_folder / "dataframe.csv"
    distance_df.to_csv(dataframe_path, index=False)

    print(
        fg.red
        + f"CSV dataframe and saved in "
        + fg.green
        + f"{str(dataframe_path)}"
        + rs.fg
    )

    return distance_df, layers_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--toml_config_path",
        "-toml",
        default=f"{os.path.dirname(__file__)}/default_distance_config.toml",
    )
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, "r") as f:
        toml_config = toml.load(f)
    print(f"**** Selected {args.toml_config_path} ****")
    compute_distance(**toml_config)
