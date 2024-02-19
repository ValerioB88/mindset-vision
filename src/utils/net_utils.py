from typing import List

import numpy as np
from torchvision.transforms import transforms

import torch.nn.functional as F
import torchvision
import torch.nn as nn
from sty import fg, ef, rs, bg
import torch.backends.cudnn as cudnn
import torch

from src.utils.callbacks import Callback, CallbackList
from src.utils.device_utils import to_global_device


class GrabNet:
    @classmethod
    def get_net(cls, architecture_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        norm_values = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        resize_value = 224

        if imagenet_pt:
            print(fg.red + "Loading ImageNet Pretraining" + rs.fg)

        pretrained_weights = "IMAGENET1K_V1" if imagenet_pt else None
        nc = 1000 if imagenet_pt else num_classes

        kwargs = dict(num_classes=nc) if nc is not None else dict()
        if architecture_name == "resnet152_decoder":
            net = ResNet152decoders(
                imagenet_pt=imagenet_pt,
                num_outputs=num_classes,
                use_residual_decoder=False,
            )
        elif architecture_name == "resnet152_decoder_residual":
            net = ResNet152decoders(
                imagenet_pt=imagenet_pt,
                num_outputs=num_classes,
                use_residual_decoder=True,
            )
        elif architecture_name == "vgg11":
            net = torchvision.models.vgg11(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features, num_classes
                )
        elif architecture_name == "vgg11bn":
            net = torchvision.models.vgg11_bn(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features, num_classes
                )
        elif architecture_name == "vgg16":
            net = torchvision.models.vgg16(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features,
                    num_classes,
                )
        elif architecture_name == "vgg16bn":
            net = torchvision.models.vgg16_bn(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features, num_classes
                )
        elif architecture_name == "vgg19bn":
            net = torchvision.models.vgg19_bn(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features,
                    num_classes,
                )
        elif architecture_name == "resnet18":
            net = torchvision.models.resnet18(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif architecture_name == "resnet50":
            net = torchvision.models.resnet50(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif architecture_name == "resnet152":
            net = torchvision.models.resnet152(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif architecture_name == "alexnet":
            net = torchvision.models.alexnet(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(
                    net.classifier[-1].in_features, num_classes
                )
        elif architecture_name == "inception_v3":  # nope
            net = torchvision.models.inception_v3(
                weights=pretrained_weights, progress=True, **kwargs
            )
            resize_value = 299
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif architecture_name == "densenet121":
            net = torchvision.models.densenet121(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif architecture_name == "densenet201":
            net = torchvision.models.densenet201(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif architecture_name == "googlenet":
            net = torchvision.models.googlenet(
                weights=pretrained_weights, progress=True, **kwargs
            )
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(architecture_name, imagenet_pt, **kwargs)
            assert (
                False if net is False else True
            ), f"Network name {architecture_name} not recognized"

        return net, norm_values, resize_value

    @staticmethod
    def get_other_nets(architecture_name, num_classes, imagenet_pt, **kwargs):
        pass


from src.utils.device_utils import GLOBAL_DEVICE


def load_pretraining(
    net=None,
    optimizers=None,
    net_state_dict_path=None,
    optimizers_path=None,
):
    if optimizers_path:
        print(
            fg.red
            + f"Loading all decoders optimizers from {optimizers_path}..."
            + rs.fg,
            end="",
        )
        opts_state = torch.load(
            optimizers_path,
            map_location=GLOBAL_DEVICE,
        )
        [opt.load_state_dict(lopt) for opt, lopt in zip(optimizers, opts_state)]
        print(fg.red + " Done." + rs.fg)

    if net_state_dict_path:
        print(
            fg.red + f"Loading full model from {net_state_dict_path}..." + rs.fg,
            end="",
        )
        net.load_state_dict(
            torch.load(
                net_state_dict_path,
                map_location=GLOBAL_DEVICE,
            )
        )
        print(fg.red + " Done." + rs.fg)


def print_net_info(net):
    num_trainable_params = 0
    tmp = ""
    print(fg.yellow)
    print("Params to learn:")
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            tmp += "\t" + name + "\n"
            print("\t" + name)
            num_trainable_params += len(param.flatten())
    print(f"Trainable Params: {num_trainable_params}")

    print("***Network***")
    print(net)
    print(
        ef.inverse
        + f"Network is in {('~train~' if net.training else '~eval~')} mode."
        + rs.inverse
    )
    print(rs.fg)
    print()


##
class Logs:
    value = None

    def __repr__(self):
        return f"{self.value}"

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value**power

    def __format__(self, format_spec):
        return format(self.value, format_spec)


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n * self.value) / (self.n + 1)
        self.n += 1
        return self


class ExpMovingAverage(Logs):
    value = None

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 - self.alpha) * self.value
        return self


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n * self.value) / (self.n + 1)
        self.n += 1
        return self


def run(
    data_loader,
    net,
    callbacks: List[Callback] = None,
    optimizer=None,
    loss_fn=None,
    iteration_step=None,
    logs=None,
    logs_prefix="",
    **kwargs,
):
    if logs is None:
        logs = {}
    torch.cuda.empty_cache()

    net = to_global_device(net)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_loss_fn(loss_fn)
    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs.update(
        {
            f"{logs_prefix}tot_iter": 0,
            f"{logs_prefix}stop": False,
            f"{logs_prefix}epoch": 0,
        }
    )
    while True:
        callbacks.on_epoch_begin(epoch, logs)
        logs[f"{logs_prefix}epoch"] = epoch
        for batch_index, data in enumerate(data_loader, 0):
            callbacks.on_batch_begin(batch_index, logs)
            iteration_step(data, net, loss_fn, optimizer, logs, logs_prefix, **kwargs)
            logs.update({"stop": False})
            logs[f"{logs_prefix}tot_iter"] += 1

            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)
            if logs[f"stop"]:
                break
            tot_iter += 1

        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs[f"stop"]:
            break

    callbacks.on_train_end(logs)
    return net, logs


class ResidualBlockPreActivation(nn.Module):
    """
    Homemade implementation of residual block with pre-activation from He et al. (2016)
    """

    def __init__(self, channels1, channels2, res_stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(channels1)
        self.conv1 = nn.Conv2d(
            channels1,
            channels2,
            kernel_size=3,
            stride=res_stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels2)
        self.conv2 = nn.Conv2d(
            channels2, channels2, kernel_size=3, stride=1, padding=1, bias=False
        )

        if res_stride != 1 or channels2 != channels1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    channels1, channels2, kernel_size=1, stride=res_stride, bias=False
                ),
                nn.BatchNorm2d(channels2),
            )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # original forward pass: Conv2d > BatchNorm2d > ReLU > Conv2D >  BatchNorm2d > ADD > ReLU
        # pre-activation forward pass: BatchNorm2d > ReLU > Conv2d > BatchNorm2d > ReLU > Conv2d > ADD
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += self.shortcut(x)

        return out


def make_layer(block, in_channel, out_channel, num_blocks, stride):
    """
    Make a layer of residual blocks
    """
    layers = []
    layers.append(block(channels1=in_channel, channels2=out_channel, res_stride=stride))
    for _ in np.arange(num_blocks - 1):
        layers.append(block(channels1=out_channel, channels2=out_channel))
    return nn.Sequential(*layers)


def replace_layer(model, old_layer, new_layer):
    for name, module in model.named_children():
        if isinstance(module, old_layer):
            setattr(model, name, new_layer())
        elif len(list(module.children())) > 0:
            replace_layer(module, old_layer, new_layer)


class ResNet152decoders(nn.Module):
    """
    ResNet152 with decoders
    """

    def __init__(
        self,
        imagenet_pt,
        num_outputs=1,
        disable_batch_norm=False,
        use_residual_decoder=False,
        **kwargs,
    ):
        super().__init__()

        pretrained_weights = "IMAGENET1K_V1" if imagenet_pt else None

        self.net = torchvision.models.resnet152(
            weights=pretrained_weights,
            progress=True,
            **kwargs,
        )

        if disable_batch_norm:
            replace_layer(self.net, nn.BatchNorm2d, nn.Identity)

        self.use_residual_decoder = use_residual_decoder

        if use_residual_decoder:
            decoder_1 = nn.Sequential(  # input: 3, 224, 224
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=3,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=64,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=64,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                # here the input will be 64, 28, 28
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_2 = nn.Sequential(  # input: 256, 56, 56
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )

            decoder_3 = nn.Sequential(  # input: 512, 28, 28
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=512,
                    out_channel=512,
                    num_blocks=2,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=512,
                    out_channel=512,
                    num_blocks=1,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )

            decoder_4 = nn.Sequential(  # input: 1024, 14, 14
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=1024,
                    out_channel=1024,
                    num_blocks=3,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(1024 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_5 = nn.Sequential(  # input: 2048, 7, 7
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=2048,
                    out_channel=2048,
                    num_blocks=3,
                    stride=1,
                ),
                nn.Flatten(),
                nn.Linear(2048 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_6 = nn.Sequential(  # input: 2048
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_outputs),
            )
            self.decoders = nn.ModuleList(
                [decoder_1, decoder_2, decoder_3, decoder_4, decoder_5, decoder_6]
            )

        else:
            self.decoders = nn.ModuleList(
                [
                    nn.Linear(3 * 224 * 224, num_outputs),
                    nn.Linear(802816, num_outputs),
                    nn.Linear(401408, num_outputs),
                    nn.Linear(200704, num_outputs),
                    nn.Linear(100352, num_outputs),
                    nn.Linear(2048, num_outputs),
                ]
            )

    def forward(self, x):
        if self.use_residual_decoder:
            return self._forward_residual_decoder(x)
        else:
            return self._forward(x)

    def _forward_residual_decoder(self, x):
        out_dec_res = []
        out_dec_res.append(self.decoders[0](x).squeeze())

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        out_dec_res.append(self.decoders[1](x))

        x = self.net.layer2(x)
        out_dec_res.append(self.decoders[2](x))

        x = self.net.layer3(x)
        out_dec_res.append(self.decoders[3](x))

        x = self.net.layer4(x)
        out_dec_res.append(self.decoders[4](x))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        out_dec_res.append(self.decoders[5](x))

        return out_dec_res

    def _forward(self, x):
        out_dec = []
        out_dec.append(self.decoders[0](torch.flatten(x, 1)))

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        out_dec.append(self.decoders[1](torch.flatten(x, 1)))

        x = self.net.layer2(x)
        out_dec.append(self.decoders[2](torch.flatten(x, 1)))

        x = self.net.layer3(x)
        out_dec.append(self.decoders[3](torch.flatten(x, 1)))

        x = self.net.layer4(x)
        out_dec.append(self.decoders[4](torch.flatten(x, 1)))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        out_dec.append(self.decoders[5](torch.flatten(x, 1)))

        return out_dec


if __name__ == "__main__":
    net = ResNet152decoders(imagenet_pt=False, num_outputs=1)
    print(net)
