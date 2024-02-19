import torch

GLOBAL_DEVICE = None


def set_global_device(device):
    """Set the global device."""
    global GLOBAL_DEVICE
    if isinstance(device, int) and torch.cuda.is_available():
        # If device is an integer, assume it's a GPU index
        GLOBAL_DEVICE = torch.device(f"cuda:{device}")
    else:
        # Otherwise, use CPU
        GLOBAL_DEVICE = torch.device("cpu")


def to_global_device(tensor_or_module):
    """Move the given tensor or module to the global device."""
    return tensor_or_module.to(GLOBAL_DEVICE)
