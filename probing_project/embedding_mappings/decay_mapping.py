import logging

import torch

from .base_mapping import MappingBase

logger = logging.getLogger(__name__)


class DecayMapping(MappingBase):
    """A class for simple contextualization of word-level embeddings.
    Computes a weighted average of the entire sentence at each word.
    """

    def __init__(self, *args, **kwargs):
        super(DecayMapping, self).__init__(*args, **kwargs)
        logger.info("Using decay mapping")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Exponential-decay contextualization of word embeddings.
        Args:
          batch: a batch of pre-computed embeddings loaded from disk.
        Returns:
          An exponentially-decaying average of the entire sequence as
          a representation for each word.
          Specifically, for word i, assigns weight:
            1 to word i
            1/2 to word (i-1,i+1)
            1/4 to word (i-2,i+2)
            ...
          before normalization by the total weight.
        """
        forward_aggregate = torch.zeros(*batch.size(), device=batch.device)
        backward_aggregate = torch.zeros(*batch.size(), device=batch.device)
        forward_normalization_tensor = torch.zeros(batch.size()[1], device=batch.device)
        backward_normalization_tensor = torch.zeros(
            batch.size()[1], device=batch.device
        )
        batch_seq_len = batch.size()[1]
        decay_constant = torch.tensor(0.5, device=batch.device)
        for i in range(batch_seq_len):
            if i == 0:
                forward_aggregate[:, i, :] = batch[:, i, :]
                backward_aggregate[:, batch_seq_len - i - 1, :] = batch[
                    :, batch_seq_len - i - 1, :
                ]
                forward_normalization_tensor[i] = 1
                backward_normalization_tensor[batch_seq_len - i - 1] = 1
            else:
                forward_aggregate[:, i, :] = (
                    forward_aggregate[:, i - 1, :] * decay_constant
                ) + batch[:, i, :]
                backward_aggregate[:, batch_seq_len - i - 1, :] = (
                    backward_aggregate[:, batch_seq_len - i, :] * decay_constant
                ) + batch[:, batch_seq_len - i - 1, :]
                forward_normalization_tensor[i] = (
                    forward_normalization_tensor[i - 1] * decay_constant + 1
                )
                backward_normalization_tensor[batch_seq_len - i - 1] = (
                    backward_normalization_tensor[batch_seq_len - i] * decay_constant
                    + 1
                )
        normalization = forward_normalization_tensor + backward_normalization_tensor
        normalization = normalization.unsqueeze(1).unsqueeze(0)
        decay_aggregate = (forward_aggregate + backward_aggregate) / normalization
        return decay_aggregate
