import logging

import torch

from .base_mapping import MappingBase

logger = logging.getLogger(__name__)


class DiskMapping(MappingBase):
    """
    A class for providing pre-computed word representations.
    Assumes the batch is constructed of loaded-from-disk embeddings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Using disk mapping")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Returns the batch itself.
        Args:
          batch: a batch of pre-computed embeddings loaded from disk.
        Returns:
          The batch, unchanged
        """
        return batch
