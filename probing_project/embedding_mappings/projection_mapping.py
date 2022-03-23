import logging

import torch
from torch import nn

from .base_mapping import MappingBase

logger = logging.getLogger(__name__)


class ProjectionMapping(MappingBase):
    """
    A class for simple contextualization of word-level embeddings.
    Runs an untrained BiLSTM on top of the loaded-from-disk embeddings.
    """

    def __init__(self, model_hidden_dim: int, *args, **kwargs):
        """
        Uses a bi-LSTM to project the sequence of embeddings to a new hidden space.
        The paramaters of the projection to the hidden space are not trained
        Args:
            model_hidden_dim: dimension of the hidden dimension that is mapped to
        """
        logger.info("Using projection mapping")
        super(ProjectionMapping, self).__init__(*args, **kwargs)
        input_dim = model_hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=int(input_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        for param in self.lstm.parameters():
            param.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Random BiLSTM contextualization of embeddings
        Args:
          batch: a batch of pre-computed embeddings loaded from disk.
        Returns:
          A random-init BiLSTM contextualization of the embeddings
        """
        with torch.no_grad():
            projected, _ = self.lstm(batch)
        return projected
