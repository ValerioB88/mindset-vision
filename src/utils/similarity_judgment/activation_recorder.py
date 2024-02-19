import glob
from itertools import product
import os
from pathlib import Path

import pandas as pd
import torch
import sty
import numpy as np
from typing import List

import PIL.Image as Image
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode, transforms
from tqdm import tqdm
from src.utils.device_utils import to_global_device

from src.utils.similarity_judgment.misc import (
    save_figs,
)
from src.utils.misc import (
    conditional_tqdm,
    conver_tensor_to_plot,
    get_affine_rnd_fun,
    my_affine,
)
from copy import deepcopy
import csv

import random


def equalize_dataframe_rows(df1, df2):
    max_length = max(len(df1), len(df2))

    # Replicate rows in df1 if it is shorter
    while len(df1) < max_length:
        df1 = pd.concat([df1, df1.sample(n=1)], ignore_index=True)

    # Replicate rows in df2 if it is shorter
    while len(df2) < max_length:
        df2 = pd.concat([df2, df2.sample(n=1)], ignore_index=True)

    return df1, df2


class RecordActivations:
    def __init__(self, net, only_save: List[str] = None, detach_tensors=True):
        if only_save is None:
            self.only_save = ["Conv2d", "Linear"]
        else:
            self.only_save = only_save
        self.cuda = False
        self.net = net
        self.detach_tensors = detach_tensors
        self.activation = {}
        self.last_linear_layer = ""
        self.all_layers_names = []
        self.setup_network()

    def setup_network(self):
        self.was_train = self.net.training
        self.net.eval()  # a bit dangerous
        print(
            sty.fg.yellow + "Network put in eval mode in Record Activation" + sty.rs.fg
        )
        all_layers = self.group_all_layers()
        self.hook_lists = []
        for idx, i in enumerate(all_layers):
            name = "{}: {}".format(idx, str.split(str(i), "(")[0])
            if np.any([ii in name for ii in self.only_save]):
                ## Watch out: not all of these layers will be used. Some networks have conditional layers depending on training/eval mode. The best way to get the right layers is to check those that are returned in "activation"
                self.all_layers_names.append(name)
                self.hook_lists.append(
                    i.register_forward_hook(self.get_activation(name))
                )
        self.last_linear_layer = self.all_layers_names[-1]

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach() if self.detach_tensors else output

        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(net):
            for layer in net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.net)
        return all_layers

    def remove_hooks(self):
        for h in self.hook_lists:
            h.remove()
        if self.was_train:
            self.net.train()


def find_matching_combinations(df1, df2, matching, not_matching):
    # Create all possible combinations of rows from both dataframes
    combinations = product(df1.iterrows(), df2.iterrows())

    # Initialize a list to store the valid combinations
    valid_combinations = []

    # Iterate over each combination
    for (idx1, row1), (idx2, row2) in combinations:
        # Check if the elements in the matching list are the same
        if all(row1[match] == row2[match] for match in matching):
            # Check if the elements in the not_matching list are different
            if all(row1[nomatch] != row2[nomatch] for nomatch in not_matching):
                # If both conditions are met, add the combination to the list
                valid_combinations.append((idx1, idx2))

    return valid_combinations


class RecordDistance(RecordActivations):
    def __init__(
        self,
        annotation_filepath,
        match_factors,
        non_match_factors,
        factor_variable,
        filter_factor_level,
        reference_level,
        distance_metric,
        *args,
        **kwargs,
    ):
        self.filter_factor_level = filter_factor_level
        self.distance_method_name = distance_metric
        self.annotation_filepath = annotation_filepath
        self.match_factors = match_factors
        self.non_match_factors = non_match_factors
        self.factor_variable = factor_variable
        self.reference_level = reference_level

        if self.distance_method_name == "cossim":
            self.distance_function = lambda im1_act, im2_act: torch.nn.CosineSimilarity(
                dim=0
            )(im1_act, im2_act).item()
        elif self.distance_method_name == "euclidean":
            self.distance_function = lambda im1_act, im2_act: torch.norm(
                (im1_act - im2_act)
            ).item()
        # elif self.distance_metric == #YOUR METRIC NAME HERE
        #     self.distance_function = lambda im1_act, im2_act: YOUR METRIC HERE
        else:
            assert False, f"{self.distance_method_name} not recognized"

        super().__init__(*args, **kwargs)

    def compute_distance_pair(self, image0, image1):
        distance = {}

        self.net(to_global_device(image0.unsqueeze(0)))
        first_image_act = {}
        for name, features1 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            first_image_act[name] = features1.flatten()

        self.net(to_global_device(image1.unsqueeze(0)))

        second_image_act = {}
        for name, features2 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            second_image_act[name] = features2.flatten()
            if name not in distance:
                distance[name] = self.distance_function(
                    first_image_act[name], second_image_act[name]
                )

        return distance

    def compute_from_annotation(
        self,
        transform,
        matching_transform=False,
        fill_bk=None,
        transf_boundaries="",
        transformed_repetition=5,
        path_save_fig=None,
        add_columns=None,
    ):
        affine_rnd_fun = get_affine_rnd_fun(transf_boundaries)
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][
            0
        ]
        df = pd.read_csv(self.annotation_filepath)

        mask = pd.Series([True] * len(df), index=df.index, dtype="bool")

        for col, val in self.filter_factor_level.items():
            mask = mask & (df[col] == val)
        df = df.loc[mask]

        all_other_levels = [
            i
            for i in [i for i in df[self.factor_variable].unique()]
            if i != self.reference_level
        ]
        pbar = tqdm(all_other_levels, desc="comparison levels")
        df_rows = []
        for comparison_level in pbar:
            pbar.set_postfix(
                {
                    sty.fg.blue
                    + f"{self.factor_variable}"
                    + sty.rs.fg: f"{self.reference_level} vs {comparison_level}"
                },
                refresh=True,
            )

            ref_df = df[df[self.factor_variable] == self.reference_level]
            comp_df = df[df[self.factor_variable] == comparison_level]
            ff = find_matching_combinations(
                ref_df, comp_df, self.match_factors, self.non_match_factors
            )

            for comb in tqdm(ff, desc="Comparison Pairs", leave=False):
                ref_row = ref_df.loc[comb[0]]
                comp_row = comp_df.loc[comb[1]]
                reference_path, comp_path = (
                    Path(self.annotation_filepath).parent / i
                    for i in [ref_row["Path"], comp_row["Path"]]
                )

                save_num_image_sets = 5

                save_sets = []
                save_fig = True

                for transform_idx in conditional_tqdm(
                    range(transformed_repetition),
                    transformed_repetition > 1,
                    desc="transformation rep.",
                    leave=False,
                ):
                    im_0 = Image.open(reference_path).convert("RGB")
                    im_i = Image.open(comp_path).convert("RGB")
                    af = (
                        [affine_rnd_fun() for i in [im_0, im_i]]
                        if not matching_transform
                        else [affine_rnd_fun()] * 2
                    )
                    images = [
                        my_affine(
                            im,
                            translate=af[idx]["tr"],
                            angle=af[idx]["rt"],
                            scale=af[idx]["sc"],
                            shear=af[idx]["sh"],
                            interpolation=InterpolationMode.NEAREST,
                            fill=fill_bk,
                        )
                        for idx, im in enumerate([im_0, im_i])
                    ]

                    images = [transform(i) for i in images]

                    layers_distances = self.compute_distance_pair(images[0], images[1])

                    df_rows.append(
                        {
                            "ReferencePath": str(reference_path),
                            "ComparisonPath": str(comp_path),
                            "ReferenceLevel": self.reference_level,
                            "ComparisonLevel": comparison_level,
                            **(
                                {f"{mf}": ref_row[mf] for mf in self.match_factors}
                                if len(self.match_factors) > 0
                                else {}
                            ),
                            **(
                                {
                                    f"{rr[0]} - {d}": rr[1][d]
                                    for rr in zip(
                                        ["Reference", "Comparison"],
                                        [ref_row, comp_row],
                                    )
                                    for d in add_columns
                                }
                            ),
                            "TransformerRep": transform_idx,
                            **layers_distances,
                        }
                    )
                    if save_fig:
                        save_sets.append(
                            [
                                conver_tensor_to_plot(i, norm.mean, norm.std)
                                for i in images
                            ]
                        )
                        if (
                            len(save_sets)
                            == min([save_num_image_sets, transformed_repetition])
                            and path_save_fig
                        ):
                            save_figs(
                                os.path.join(
                                    path_save_fig,
                                    f"[{self.reference_level}]_{comparison_level}"
                                    + (
                                        f"_{'-'.join([str(i) for i in self.match_factors])}"
                                        if len(self.match_factors) > 0
                                        else ""
                                    ),
                                ),
                                save_sets,
                                extra_info=transf_boundaries,
                            )

                            save_fig = False
                            save_sets = []
        results_df = pd.DataFrame(df_rows)

        all_layers = list(layers_distances.keys())
        return results_df, all_layers
