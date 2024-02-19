import os
import pathlib
import pickle
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision
import random
from PIL import ImageStat
from sty import fg, rs
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.transforms import functional as F
from sklearn.preprocessing import MinMaxScaler


def add_compute_stats(obj_class):
    class ComputeStatsUpdateTransform(obj_class):
        # This class basically is used for normalize Dataset Objects such as ImageFolder in order to be used in our more general framework
        def __init__(
            self,
            name_ds="dataset",
            add_PIL_transforms=None,
            add_tensor_transforms=None,
            num_image_calculate_mean_std=70,
            stats=None,
            save_stats_file=None,
            **kwargs,
        ):
            """

            @param add_tensor_transforms:
            @param stats: this can be a dict (previous stats, which will contain 'mean': [x, y, z] and 'std': [w, v, u],
                          a str "ImageNet", indicating the ImageNet stats,
                          a str pointing to a path to a pickle file, containing a dict with 'mean' and 'std'
                          None, indicating that stats are gonna be computed
                        In any case, the stats are gonna be added as a normalizing transform.
            @param save_stats_file: a path, indicating where to save the stats
            @param kwargs:
            """
            self.verbose = True
            print(
                fg.yellow
                + f"\n**Creating Dataset ["
                + fg.cyan
                + f"{name_ds}"
                + fg.yellow
                + "]**"
                + rs.fg
            )
            super().__init__(**kwargs)
            # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose(
                [
                    *add_PIL_transforms,
                    torchvision.transforms.ToTensor(),
                    *add_tensor_transforms,
                ]
            )

            self.name_ds = name_ds
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std

            compute_stats = False

            if isinstance(stats, dict):
                self.stats = stats
                print(
                    fg.red
                    + f"Using precomputed stats: "
                    + fg.cyan
                    + f"mean = {self.stats['mean']}, std = {self.stats['std']}"
                    + rs.fg
                )

            elif stats == "ImageNet":
                self.stats = {}
                self.stats["mean"] = [0.491, 0.482, 0.44]
                self.stats["std"] = [0.247, 0.243, 0.262]
                print(
                    fg.red
                    + f"Using ImageNet stats: "
                    + fg.cyan
                    + f"mean = {self.stats['mean']}, std = {self.stats['std']}"
                    + rs.fg
                )

            elif isinstance(stats, str):
                if os.path.isfile(stats):
                    self.stats = pickle.load(open(stats, "rb"))
                    print(
                        fg.red
                        + f"Using stats from file [{Path(stats).name}]: "
                        + fg.cyan
                        + f"mean = {self.stats['mean']}, std = {self.stats['std']}"
                        + rs.fg
                    )
                    if stats == save_stats_file:
                        save_stats_file = None
                else:
                    print(
                        fg.red
                        + f"File [{Path(stats).name}] not found, stats will be computed."
                        + rs.fg
                    )
                    compute_stats = True

            if stats is None or compute_stats is True:
                self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"Stats saved in {save_stats_file}")
                pathlib.Path(os.path.dirname(save_stats_file)).mkdir(
                    parents=True, exist_ok=True
                )
                pickle.dump(self.stats, open(save_stats_file, "wb"))

            normalize = torchvision.transforms.Normalize(
                mean=self.stats["mean"], std=self.stats["std"]
            )

            self.transform.transforms += [normalize]

        def call_compute_stats(self):
            return compute_mean_and_std_from_dataset(
                self,
                None,
                max_iteration=self.num_image_calculate_mean_std,
                verbose=self.verbose,
            )

    return ComputeStatsUpdateTransform


class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, np.array(self.h) / 255, np.array(other.h) / 255)))


def compute_mean_and_std_from_dataset(
    dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True
):
    if max_iteration < 30:
        print(
            "Max Iteration in Compute Mean and Std for dataset is lower than 30! This could create unrepresentative stats!"
        ) if verbose else None
    start = time()
    idxs = random.choices(range(len(dataset)), k=np.min((max_iteration, len(dataset))))
    imgs = [dataset[i][0] for i in idxs]  # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()
    means = [imgs[:, i, :, :].mean() for i in range(3)]
    stds = [imgs[:, i, :, :].std(ddof=0) for i in range(3)]
    stats = {
        "mean": means,
        "std": stds,
        "time_one_iter": (time() - start) / max_iteration,
        "iter": max_iteration,
    }

    print(
        fg.cyan
        + f'mean={np.around(stats["mean"],4)}, std={np.around(stats["std"], 4)}, time1it: {np.around(stats["time_one_iter"], 4)}s'
        + rs.fg
    ) if verbose else None
    if dataset_path is not None:
        print("Saving in {}".format(dataset_path))
        with open(dataset_path, "wb") as f:
            pickle.dump(stats, f)

    return stats


def fix_dataset(dataset, transf_values, fill_color, name_ds=""):
    dataset.name = name_ds
    dataset.stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }  # ImageNet Normalization Values
    add_resize = False
    if next(iter(dataset))[0].size[0] != 244:
        add_resize = True

    dataset.transform = torchvision.transforms.Compose(
        [
            AffineTransform(
                transf_values["translation_X"]
                if not transf_values["translation_X"] is False
                else (0, 0),
                transf_values["translation_Y"]
                if not transf_values["translation_Y"] is False
                else (0, 0),
                transf_values["rotation"]
                if not transf_values["rotation"] is False
                else (0, 0),
                transf_values["scale"]
                if not transf_values["scale"] is False
                else (1, 1),
                fill_color=fill_color,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=dataset.stats["mean"], std=dataset.stats["std"]
            ),
        ]
    )
    if add_resize:
        dataset.transform.transforms.insert(0, torchvision.transforms.Resize(224))
    return dataset


def get_dataloader(
    task_type,
    ds_config,
    transf_config,
    batch_size,
    return_path=False,
):
    ds = ImageDatasetAnnotations(
        task_type=task_type,
        csv_file=ds_config["annotation_file"],
        img_path_col=ds_config["img_path_col_name"],
        label_cols=ds_config["label_cols"],
        filters=ds_config["filters"],
        transform=None,  # transform is added in fix_dataset
        return_path=return_path,
    )

    ds = fix_dataset(
        ds,
        transf_values=transf_config["values"],
        fill_color=transf_config["fill_color"],
        name_ds=ds_config["name"],
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8 if torch.cuda.is_available() else 0,
        timeout=0,
        pin_memory=True,
    )


from torch.utils.data import Dataset
import pandas as pd

import json
from torchvision.models import ResNet50_Weights

ResNet50_Weights.IMAGENET1K_V1.value.meta["categories"]

import pandas as pd
import json


class ImageNetClasses:
    def __init__(self, class_index_json_file="assets/imagenet_class_index.json"):
        self.class_idx = json.load(open(class_index_json_file))
        self.idx2label = [self.class_idx[str(k)][1] for k in range(len(self.class_idx))]
        self.label2idx = {
            names[1]: int(label) for label, names in self.class_idx.items()
        }

    def add_to_annotation_file_path(
        self, annotation_file_path, name_cols_imagenet, annotation_output_path=None
    ):
        """
        This function adds a column to the annotation file, with the ImageNet class index.
        """
        annotation_output_path = (
            annotation_file_path
            if annotation_output_path is None
            else annotation_output_path
        )
        df = pd.read_csv(annotation_file_path)

        def fun(class_name):
            return (
                self.label2idx[class_name] if class_name in self.label2idx else "none"
            )

        df["ImageNetClassIdx"] = df[name_cols_imagenet].apply(fun)
        df.to_csv(annotation_output_path)
        print(
            f"ImageNet class added in columns ImageNetClassIdx based on the class {name_cols_imagenet}"
        )


class ImageDatasetAnnotations(Dataset):
    def __init__(
        self,
        task_type: str,
        csv_file: str,
        img_path_col: str,
        label_cols: Union[List[str], str],
        filters: Optional[Dict[str, Union[str, int]]] = None,
        transform=None,
        return_path=False,
    ):
        self.return_path = return_path
        self.task_type = task_type
        self.dataframe = pd.read_csv(csv_file)
        if filters:
            for key, value in filters.items():
                self.dataframe = self.dataframe[self.dataframe[key] == value]
        self.img_path_col = img_path_col
        self.root_path = pathlib.Path(csv_file).parent
        self.label_cols = label_cols

        if isinstance(self.label_cols, str):
            self.label_cols = [self.label_cols]

        if self.task_type == "classification":
            assert (
                len(self.label_cols) == 1
            ), "With a classification task, the dataset.label_cols must be a single string or one-element list"
            self.label_cols = self.label_cols[0]
            self.classes = self.dataframe[self.label_cols].unique()
            self.classes_dict = {
                c: i for i, c in enumerate(self.classes)
            }  # for __getitem__

        self.transform = transform
        self.dataframe = self.dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.root_path / self.dataframe.loc[idx, self.img_path_col]

        if self.task_type == "classification":
            labels = self.dataframe.loc[idx, self.label_cols]
            labels = self.classes_dict[labels]
            label_tensor_dtype = torch.long
        else:
            labels = self.dataframe.loc[idx, self.label_cols].values.astype(float)
            label_tensor_dtype = torch.float32

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.return_path:
            return image, torch.tensor(labels, dtype=label_tensor_dtype), str(img_path)
        else:
            return image, torch.tensor(labels, dtype=label_tensor_dtype)


class AffineTransform:
    def __init__(
        self, translate_X, translate_Y, rotate, scale, fill_color=None
    ) -> None:
        self.translate_X = translate_X
        self.translate_Y = translate_Y
        self.rotate = rotate
        self.scale = scale
        self.fill_color = [0, 0, 0] if fill_color is None else fill_color

    def __call__(self, img):
        return F.affine(
            img,
            angle=random.uniform(*self.rotate),
            translate=(
                random.uniform(*self.translate_X),
                random.uniform(*self.translate_Y),
            ),
            scale=random.uniform(*self.scale),
            shear=0,
            fill=self.fill_color,
        )
