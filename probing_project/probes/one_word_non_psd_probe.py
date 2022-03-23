import logging

import torch
import torch.nn as nn

from .probe_base import ProbeBase

logger = logging.getLogger(__name__)


class OneWordNonPSDProbe(ProbeBase):
    """
    Computes a bilinear affinity between each word representation and itself.
    This is different from the probes in A Structural Probe... as the
    matrix in the quadratic form is not guaranteed positive semi-definite
    """

    def __init__(self, model_hidden_dim: int):
        logging.info("Constructing OneWordNonPSDProbe")
        super(OneWordNonPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_hidden_dim, model_hidden_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """
        Computes all n depths after projection for each sentence in a batch.
        Computes (h_i^T)A(h_i) for all i
        Args:
            batch: a batch of word representations of the shape (batch_size, max_seq_len, representation_dim)
        Returns: A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = batch.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            batch.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms
