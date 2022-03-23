import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""

    def __init__(self):
        super(L1DistanceLoss, self).__init__()
        logger.info("Using L1 Distance Loss")
        self.word_pair_dims = (1, 2)

    def forward(
        self,
        predictions: torch.Tensor,
        label_batch: torch.Tensor,
        length_batch: torch.Tensor,
    ):
        """
        Computes L1 loss on distance matrices. Ignores all entries where label_batch=-1
        Normalizes first within sentences (dividing by the square of sentence length)
        and then across the batch.

        Args:
            predictions: A pytorch batch of predicted distances
            label_batch: A pytorch batch of true distances
            length_batch: A pytorch batch of sentence lengths

        Returns:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims
            )
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=predictions.device)
        return batch_loss, total_sents
