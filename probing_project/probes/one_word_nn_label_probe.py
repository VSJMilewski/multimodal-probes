import argparse
import logging
from typing import List

import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class OneWordNNLabelProbe(ProbeBase):
    """
    Computes an MLP function of pairs of vectors.
    For a batch of sentences, computes all n scores for each sentence in the batch.
    Used to predict a label for each word, such as POS tags.
    """

    def __init__(
        self,
        model_hidden_dim: int,
        probe_rank: int,
        label_space_size: int,
        probe_hidden_layers: int,
        probe_dropout: float,
    ):
        logging.info("Constructing OneWordNNLabelProbe")
        super(OneWordNNLabelProbe, self).__init__()
        intermediate_layers: List[nn.Module] = []
        for i in range(probe_hidden_layers - 1):
            intermediate_layers.append(nn.Linear(probe_rank, probe_rank))
            intermediate_layers.append(nn.ReLU())
            intermediate_layers.append(nn.Dropout(p=probe_dropout))
        self.nn_probe = nn.Sequential(
            nn.Dropout(p=probe_dropout),
            nn.Linear(model_hidden_dim, probe_rank),
            nn.ReLU(),
            *intermediate_layers,
            nn.Linear(probe_rank, label_space_size)
        )

    def forward(self, batch):
        """
        Computes all n label logits for each sentence in a batch.
        Computes W2(relu(W1[h_i]+b1)+b2 or
                 W3(relu(W2(relu(W1[h_i]+b1)+b2)+b3
        for MLP-1, MLP-2, respectively for all i
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        batch = self.nn_probe(batch)
        return batch

    @staticmethod
    def add_probe_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ProbeArgs")
        parser.add_argument(
            "--probe_rank",
            type=int,
            default=512,
            help="Dimension the probe maps embeddings to.",
        )
        parser.add_argument(
            "--probe_hidden_layers",
            type=int,
            default=2,
            help="Number of laers in probe.",
        )
        parser.add_argument(
            "--probe_dropout",
            type=float,
            default=0.5,
            help="dropout percentage in probe.",
        )
        return parent_parser
