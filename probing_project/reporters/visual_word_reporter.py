import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch

from .word_reporter import WordReporter

logger = logging.getLogger(__name__)


class VisualWordReporter(WordReporter):
    """Reporting class for single-word (depth) tasks for the image objects"""

    def report_root_acc(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[Any],
        split_name: str,
    ) -> Dict[str, float]:
        """
        Computes the root prediction accuracy and writes to disk.
        For each sentence in the corpus, the root token in the sentence
        should be the least deep. This is a simple evaluation.
        Computes the percentage of sentences for which the root token
        is the least deep according to the predicted depths; writes
        this value to disk.

        Args:
          predictions: A sequence of batches of predictions for a data split
          labels:
          lengths:
          observations: A sequence of batches of Observations
          split_name the string naming the data split: {train,dev,test}
        """
        total_sents = 0
        correct_root_predictions = 0
        for prediction_batch, label_batch, length_batch, observation_batch in zip(
            predictions, labels, lengths, observations
        ):
            for idx in range(prediction_batch.size(0)):
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                labels_1s = (
                    label_batch[idx, :length] > self.mask_max_value
                )  # all labels higher than mask_value
                label_0_index = (
                    (label_batch[idx, :length][labels_1s].detach() == 0)
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                prediction = (
                    prediction_batch.data[idx, :length][labels_1s].detach().cpu()
                )
                correct_root_predictions += label_0_index == np.argmin(prediction)
                total_sents += 1
        root_acc = correct_root_predictions / float(total_sents)
        with open(
            os.path.join(self.reporting_root, split_name + ".root_acc"), "w"
        ) as f:
            f.write(
                "\t".join(
                    [str(root_acc), str(correct_root_predictions), str(total_sents)]
                )
                + "\n"
            )
        return {"root_acc": root_acc}
