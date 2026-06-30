import os
import threading
from contextlib import nullcontext

import torch

_MPS_LOCK = threading.RLock()


def mps_serialized(device: torch.device | str | None):
    if os.environ.get("LADA_SERIALIZE_MPS", "1") != "1":
        return nullcontext()
    if device is None:
        return nullcontext()
    torch_device = torch.device(device)
    if torch_device.type != "mps":
        return nullcontext()
    return _MPS_LOCK
