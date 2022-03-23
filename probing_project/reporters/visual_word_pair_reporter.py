import logging
import os
from typing import Any, Dict, List

import torch
from probing_project.utils import dist_matrix_to_edges

from .word_pair_reporter import WordPairReporter

logger = logging.getLogger(__name__)


class VisualWordPairReporter(WordPairReporter):
    """Reporting class for single-word (depth) tasks for the image objects"""

    def report_uuas(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[Any],
        split_name: str,
    ) -> Dict[str, float]:
        """
        Computes the UUAS score for a dataset.
        From the true and predicted distances, computes a minimum spanning tree
        of each, and computes the percentage overlap between edges in all
        predicted and gold trees.
        All tokens with punctuation part-of-speech are excluded from the minimum
        spanning trees.

        Args:
            predictions: A sequence of batches of predictions for a data split
            labels:
            lengths:
            observations: A sequence of batches of Observations
            split_name: the string naming the data split: {train,dev,test}
        """
        uspan_total = 0
        uspan_correct = 0
        total_sents = 0
        for prediction_batch, label_batch, length_batch, observation_batch in zip(
            predictions, labels, lengths, observations
        ):
            for idx in range(prediction_batch.size()[0]):
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                # we use the max mask value to only use the pairs from
                # both predictions and labels for needed regions
                labels_1s = (
                    label_batch[idx, :length, :length] > self.mask_max_value
                )  # all labels higher than mask_value
                num_regs = sum(labels_1s[0])
                if num_regs == 0:
                    continue
                prediction = (
                    prediction_batch[idx, :length, :length][labels_1s]
                    .view(num_regs, num_regs)
                    .detach()
                    .cpu()
                )
                label = (
                    label_batch[idx, :length, :length][labels_1s]
                    .view(num_regs, num_regs)
                    .detach()
                    .cpu()
                )

                gold_edges = dist_matrix_to_edges(label)
                pred_edges = dist_matrix_to_edges(prediction)

                uspan_correct += len(
                    set([tuple(sorted(x)) for x in gold_edges]).intersection(
                        set([tuple(sorted(x)) for x in pred_edges])
                    )
                )
                uspan_total += len(gold_edges)
                total_sents += 1
        uuas = uspan_correct / float(uspan_total)
        with open(os.path.join(self.reporting_root, split_name + ".uuas"), "w") as f:
            f.write(str(uuas) + "\n")
        return {"uuas": uuas}
