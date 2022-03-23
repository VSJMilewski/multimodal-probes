import logging

import torch
import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class TwoWordNonPSDProbe(ProbeBase):
    """
    Computes a bilinear function of difference vectors.
    For batch of sentences, computes all n^2 pairs of scores for each sentence in batch.
    Can be used to probe for the distances between words in a tree.
    """

    def __init__(self, model_hidden_dim: int):
        logging.info("Constructing TwoWordNonPSDProbe")
        super(TwoWordNonPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_hidden_dim, model_hidden_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """
        Computes all n^2 pairs of difference scores for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (h_i-h_j)^TA(h_i-h_j) for all i,j
        Args:
            batch: a batch of word representations of the shape
                   (batch_size, max_seq_len, representation_dim)
        Returns: A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = batch.size()
        batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
        diffs = (batch_square - batch_square.transpose(1, 2)).view(
            batchlen * seqlen * seqlen, rank
        )
        psd_transformed = torch.matmul(diffs, self.proj).view(
            batchlen * seqlen * seqlen, 1, rank
        )
        dists = torch.bmm(
            psd_transformed, diffs.view(batchlen * seqlen * seqlen, rank, 1)
        )
        dists = dists.view(batchlen, seqlen, seqlen)
        return dists
