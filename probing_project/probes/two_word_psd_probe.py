import argparse
import logging

import torch
import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class TwoWordPSDProbe(ProbeBase):
    """
    Computes squared L2 distance after projection by a matrix.
    For batch of sentences, computes all pairs of distances for each sentence in batch.
    Can be used to probe for the distances between words in a tree.
    """

    def __init__(self, model_hidden_dim: int, probe_rank: int):
        logging.info("Constructing TwoWordPSDProbe")
        super(TwoWordPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_hidden_dim, probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """
        Computes all n^2 pairs of distances after projection for each sentence in batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
            batch: a batch of word representations of the shape
                   (batch_size, max_seq_len, representation_dim)
        Returns: A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances

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
        return parent_parser
