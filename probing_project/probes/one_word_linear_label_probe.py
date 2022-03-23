import argparse
import logging

import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class OneWordLinearLabelProbe(ProbeBase):
    """
    Computes a linear function of pairs of vectors.
    For a batch of sentences, computes all n scores for each sentence in the batch.
    Used to predict a label for each word, such as POS tags.
    """

    def __init__(
        self,
        model_hidden_dim: int,
        probe_rank: int,
        label_space_size: int,
        probe_dropout: float,
    ):
        logging.info("Constructing OneWordLinearLabelProbe")
        super(OneWordLinearLabelProbe, self).__init__()
        self.dropout = nn.Dropout(p=probe_dropout)
        self.nn_probe = nn.Sequential(
            nn.Dropout(p=probe_dropout),
            nn.Linear(model_hidden_dim, probe_rank),
            nn.Linear(probe_rank, label_space_size),
        )

    def forward(self, batch):
        """
        Computes all n label logits for each sentence in a batch.
        Computes W2(W1(h_i+b1)+b2 for all i
        why the two steps? Because
              W1 in R^{maximum_rank x hidden_dim}, W2 in R^{hidden_dim, maximum_rank}
        this rank constraint enforces a latent linear space of rank
        maximum_rank or less.
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        logits = self.nn_probe(batch)
        return logits

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
            help="Number of layers in probe.",
        )
        parser.add_argument(
            "--probe_dropout",
            type=float,
            default=0.5,
            help="dropout percentage in probe.",
        )
        return parent_parser
