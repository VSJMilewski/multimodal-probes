import argparse
import logging
from typing import List

import torch
import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class OneWordNNProbe(ProbeBase):
    """
    Computes all squared L2 norm of n words as depths after an MLP projection.
    Can be used for probing the depth of words in a tree
    """

    def __init__(
        self, model_hidden_dim: int, probe_hidden_layers: int, intermediate_dim: int
    ):
        logging.info("Constructing OneWordNNDepthProbe")
        super(OneWordNNProbe, self).__init__()
        initial_linear = nn.Linear(model_hidden_dim, intermediate_dim)
        intermediate_layers: List[nn.Module] = []
        for i in range(probe_hidden_layers):
            intermediate_layers.append(nn.Linear(intermediate_dim, intermediate_dim))
            if i != probe_hidden_layers - 1:
                intermediate_layers.append(nn.ReLU())
        self.nn_probe = nn.Sequential(initial_linear, nn.ReLU(), *intermediate_layers)

    def forward(self, batch):
        """
        Computes all squared L2 norm of n words as depths after an MLP projection
         for each sentence in a batch. predicts the depth through an MLP
        Args:
            batch: a batch of word representations of the shape
                   (batch_size, max_seq_len, representation_dim)
        Returns: A tensor of depths of shape (batch_size, max_seq_len)
        """
        batch = self.nn_probe(batch)
        batchlen, seqlen, rank = batch.size()
        norms = torch.bmm(
            batch.view(batchlen * seqlen, 1, rank),
            batch.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms

    @staticmethod
    def add_probe_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ProbeArgs")
        parser.add_argument(
            "--probe_hidden_layers",
            type=int,
            default=2,
            help="Number of laers in probe.",
        )
        parser.add_argument(
            "--intermediate_dim",
            type=int,
            default=300,
            help="Dimension the probe maps embeddings to.",
        )
        return parent_parser
