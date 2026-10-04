import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_batch_size() -> int:
    """A100(80GB) -> 64, >=16GB -> 32, >=8GB -> 16, CPU or <8GB -> 8."""
    if not torch.cuda.is_available():
        return 8
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 40:
        return 64
    if total_gb >= 16:
        return 32
    if total_gb >= 8:
        return 16
    return 8


def gpu_info_str() -> str:
    if not torch.cuda.is_available():
        return "No GPU detected (CPU only)"
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / (1024 ** 3)
    return f"{props.name} ({gb:.1f} GB)"
