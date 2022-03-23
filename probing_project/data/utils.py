import logging
from collections.abc import Mapping, Sequence

import torch
from probing_project.utils import Observation
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
    string_classes,
)

logger = logging.getLogger(__name__)


def updated_default_collate(batch):
    """
    COPIED AND ADAPTED FROM
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    Puts each data field into a tensor with outer dimension batch size.
    Changed named tuple part of code
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return updated_default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, Mapping):
        return {key: updated_default_collate([d[key] for d in batch]) for key in elem}
    # NOTE<  NAMEDTUPLE CODE IS CHANGED!
    elif isinstance(elem, Observation):  # namedtuple
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(updated_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [updated_default_collate(samples) for samples in transposed]


def filter_none_from_updated_default_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return updated_default_collate(batch)
