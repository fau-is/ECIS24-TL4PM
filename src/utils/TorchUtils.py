import torch
import importlib.util


def get_torch_device():
    torch_device = "cpu"
    device_package = torch.cpu
    if importlib.util.find_spec("torch.backends.mps") is not None:
        if torch.backends.mps.is_available():
            torch_device = torch.device("mps")
            device_package = torch.mps
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        device_package = torch.cuda
    return torch_device, device_package
