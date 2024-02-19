import torch
from torch import nn as nn
from src.utils.device_utils import to_global_device
from src.utils.misc import convert_lists_to_strings
from copy import deepcopy


def update_logs(
    logs, loss_decoders, output_decoders, labels, method, logs_prefix, train=True
):
    logs[f"{logs_prefix}ema_loss"].add(sum(loss_decoders))
    prefix = "ema_" if train else ""
    if method == "regression":
        for idx, ms in enumerate(loss_decoders):
            logs[f"{logs_prefix}{prefix}rmse_{idx}"].add(torch.sqrt(ms).item())
        if not train:
            logs[f"{logs_prefix}rmse"].add(
                torch.sqrt(sum(loss_decoders) / len(loss_decoders)).item()
            )
    elif method == "classification":
        for idx in range(len(loss_decoders)):
            acc = torch.mean(
                (torch.argmax(output_decoders[idx], 1) == labels).float()
            ).item()
            logs[f"{logs_prefix}{prefix}acc_{idx}"].add(acc)
        average_acc = torch.mean(
            torch.tensor(
                [
                    logs[f"{logs_prefix}{prefix}acc_{idx}"].value
                    for idx in range(len(loss_decoders))
                ]
            )
        ).item()
        if not train:
            logs[f"{logs_prefix}acc"].add(average_acc)


# import torchvision
# p = torchvision.transforms.transforms.ToPILImage()(images[0])


def decoder_step(
    data,
    model,
    loss_fn,
    optimizers,
    logs,
    logs_prefix,
    train,
    method,
    **kwargs,
):
    num_decoders = len(model.decoders)

    images, labels = data
    images = to_global_device(images)
    labels = to_global_device(labels)
    if train:
        [optimizers[i].zero_grad() for i in range(num_decoders)]
    out_dec = model(images)
    loss = to_global_device(torch.tensor([0.0], dtype=torch.float, requires_grad=True))
    loss_decoder = []
    for _, od in enumerate(out_dec):
        loss_decoder.append(loss_fn(od, labels))
        loss = loss + loss_decoder[-1]

    update_logs(logs, loss_decoder, out_dec, labels, method, logs_prefix, train)

    if "collect_data" in kwargs and kwargs["collect_data"]:
        logs["data"] = data

    if train:
        loss.backward()
        [optimizers[i].step() for i in range(num_decoders)]


def log_neptune_init_info(neptune_logger, toml_config, tags=None):
    tags = [] if tags is None else tags
    neptune_logger["sys/name"] = toml_config["train_info"][
        "run_id"
    ]  # to be consistent with before
    neptune_logger["toml_config"] = convert_lists_to_strings(toml_config)
    neptune_logger["toml_config_file"].upload(
        toml_config["train_info"]["save_folder"] + "/toml_config.txt"
    )
    neptune_logger["sys/tags"].add(tags)


def replace_layer(net, layer_class, new_layer_class):
    """This function replaces a specific layer class in a given neural network with a new layer class. The input parameters are:
    net: The neural network object in which the layer replacement is to be done
    layer_class: The class of the layer to be replaced
    new_layer_class: The class of the new layer to replace the old one
    """

    layers = deepcopy(
        list(net.named_modules())
    )  # copy to avoid RuntimeError: dictionary changed size during iteration
    for name, layer in layers:
        if isinstance(layer, layer_class):
            names = name.split(".")
            parent = net
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], new_layer_class())
