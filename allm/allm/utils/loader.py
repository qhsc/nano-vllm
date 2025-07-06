import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm.auto import tqdm

from allm.utils.dist import get_tp_rank


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    with tqdm(
        glob(os.path.join(path, "*.safetensors")),
        desc="Loading model weights",
        disable=get_tp_rank() != 0,
    ) as pbar:
        for file in pbar:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    for origin_key in packed_modules_mapping:
                        if origin_key in weight_name:
                            packed_name, shard_id = packed_modules_mapping[origin_key]
                            param_name = weight_name.replace(origin_key, packed_name)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:  # not packed
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, f.get_tensor(weight_name))
