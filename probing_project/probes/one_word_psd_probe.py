import argparse
import logging

import torch
import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class OneWordPSDProbe(ProbeBase):
    """
    Computes squared L2 norm of words after projection by a matrix.
    Can be used for probing the depth of words in a tree
    """

    def __init__(self, model_hidden_dim: int, probe_rank: int):
        logging.info("Constructing OneWordPSDProbe")
        super(OneWordPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_hidden_dim, probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """
        Computes all n depths after projection for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i
        Args:
            batch: a batch of word representations of the shape
                   (batch_size, max_seq_len, representation_dim)
        Returns: A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            transformed.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms

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
