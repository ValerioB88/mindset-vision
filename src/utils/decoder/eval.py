import json
import os
import torch
import torch.backends.cudnn as cudnn

# from rich import print
from src.utils.callbacks import *
from src.utils.dataset_utils import get_dataloader
from src.utils.device_utils import set_global_device, to_global_device
from src.utils.misc import pretty_print_dict, update_dict
from src.utils.net_utils import load_pretraining, GrabNet
from tqdm import tqdm
import pandas
import toml
import inspect

import inspect


def decoder_evaluate(
    task_type=None, gpu_idx=None, eval=None, network=None, saving_folders=None, **kwargs
):
    with open(os.path.dirname(__file__) + "/default_decoder_config.toml", "r") as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {i: local_vars[i] for i in inspect.getfullargspec(decoder_evaluate)[0]},
    )
    pretty_print_dict(toml_config, name="PARAMETERS")
    set_global_device(toml_config["gpu_idx"])

    test_loaders = [
        get_dataloader(
            toml_config=toml_config,
            task_type=toml_config["task_type"],
            ds_config=i,
            transf_config=toml_config["transformation"],
            batch_size=toml_config["network"]["batch_size"],
            return_path=True,
        )
        for i in toml_config["eval"]["datasets"]
    ]

    assert toml_config["network"]["architecture_name"] in [
        "resnet152_decoder",
        "resnet152_decoder_residual",
    ], f"Network.name needs to be either `resnet152_decoder` or `resnet152_decoder_residual`. You used {toml_config['network']['name']}"

    label_cols = toml_config["training"]["dataset"]["label_cols"]
    if toml_config["task_type"] == "regression":
        num_classes = (
            1
            if isinstance(label_cols, str)
            else len(toml_config["training"]["dataset"]["label_cols"])
        )
    elif toml_config["task_type"] == "classification":
        num_classes = (
            1 if isinstance(label_cols, str) else len(test_loaders[0].dataset.classes)
        )

    net, _, _ = GrabNet.get_net(
        toml_config["network"]["architecture_name"],
        imagenet_pt=True if toml_config["network"]["imagenet_pretrained"] else False,
        num_classes=num_classes,
    )

    load_pretraining(
        net=net,
        optimizers=None,
        net_state_dict_path=toml_config["network"]["state_dict_path"],
        optimizers_path=None,
    )
    net.eval()

    net = to_global_device(net)
    results_folder = pathlib.Path(toml_config["saving_folders"]["results_folder"])
    num_decoders = len(net.decoders)

    def evaluate_one_dataloader(dataloader):
        results_final = []
        (results_folder / dataloader.dataset.name).mkdir(parents=True, exist_ok=True)

        print(
            f"Evaluating Dataset "
            + sty.fg.green
            + f"{dataloader.dataset.name}"
            + sty.rs.fg
        )

        for _, data in enumerate(tqdm(dataloader, colour="yellow")):
            images, labels, path = data
            images = to_global_device(images)
            labels = to_global_device(labels)
            out_dec = net(images)
            for i in range(len(labels)):
                results_final.append(
                    {
                        "image_path": path[i],
                        "label": labels[i].item(),
                        **{
                            f"prediction_dec_{dec_idx}": torch.argmax(
                                out_dec[dec_idx][i]
                            ).item()
                            if task_type == "classification"
                            else out_dec[dec_idx][i].item()
                            for dec_idx in range(num_decoders)
                        },
                    }
                )

        results_final_pandas = pandas.DataFrame(results_final)
        result_path = str(results_folder / dataloader.dataset.name / "predictions.csv")
        results_final_pandas.to_csv(
            result_path,
            index=False,
        )
        print(sty.fg.yellow + f"Result written in {result_path}" + sty.rs.fg)

    [evaluate_one_dataloader(dataloader) for dataloader in test_loaders]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--toml_config_path", "-tomlf")
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, "r") as f:
        toml_config = toml.load(f)
    print(f"**** Selected {args.toml_config_path} ****")
    decoder_evaluate(**toml_config)
