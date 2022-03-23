import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class L1DepthLoss(nn.Module):
    """Custom L1 loss for depth sequences."""

    def __init__(self, mask_max_value=-1):
        super(L1DepthLoss, self).__init__()
        logger.info("Using L1 Depth Loss")
        self.word_dim = 1
        self.mask_value = mask_max_value

    def forward(
        self,
        predictions: torch.Tensor,
        label_batch: torch.Tensor,
        length_batch: torch.Tensor,
    ):
        """
        Computes L1 loss on depth sequences. Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.

        Args:
            predictions: A pytorch batch of predicted depths
            label_batch: A pytorch batch of true depths
            length_batch: A pytorch batch of sentence lengths

        Returns:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        total_sents = torch.sum(
            length_batch != 0
        ).float()  # count number of sentences where length is not 0
        labels_1s = (label_batch > self.mask_value).float()  # all labels of 0 or higher
        predictions_masked = (
            predictions * labels_1s
        )  # set predictions to zero, where label is -1
        labels_masked = label_batch * labels_1s  # set labels to zero, where label is -1
        if total_sents > 0:
            # sum the absolute mistake from all tokens in a sentence,
            # counting all masked labels as 0 difference.
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=self.word_dim
            )
            # devide every sentence by the length of the sentence
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            # take the average mistake from all sentences
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=predictions.device)
        return batch_loss, total_sents
