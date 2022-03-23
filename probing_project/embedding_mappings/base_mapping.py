import logging

import torch

logger = logging.getLogger(__name__)


class MappingBase(torch.nn.Module):
    """
    An abstract class for neural models that
    assign a single vector to each word in a text.
    """

    def __init__(self, *_, **__):
        super(MappingBase, self).__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Assigns a vector to each word in a batch."""
        raise NotImplementedError(
            "Model is an abstract class; use one of the implementing classes."
        )
