import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0, "numerator must be divisible by denominator"
    return numerator // denominator


_TP_GROUP = None


def get_tp_group():
    global _TP_GROUP
    if _TP_GROUP is None:
        _TP_GROUP = dist.new_group()
    return _TP_GROUP


def get_tp_size():
    return dist.get_world_size() or 1


def get_tp_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
