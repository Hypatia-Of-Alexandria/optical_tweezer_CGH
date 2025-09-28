import torch

__all__ = [
    'get_device'
]

def get_device() -> torch.device:
    """
    Return best available torch device: MPS, CUDA, else CPU.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
