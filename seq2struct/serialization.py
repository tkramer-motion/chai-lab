import json
from pathlib import PosixPath

import torch


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, PosixPath):
            return obj.as_posix()
        return json.JSONEncoder.default(self, obj)
