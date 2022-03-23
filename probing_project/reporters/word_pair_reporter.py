import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import torch
from matplotlib import pyplot as plt  # type: ignore
from probing_project.utils import dist_matrix_to_edges
from scipy.stats import spearmanr  # type: ignore

from .reporter_base import ReporterBase

logger = logging.getLogger(__name__)


class WordPairReporter(ReporterBase):
    """Reporting class for wordpair (distance) tasks"""

    def __init__(
        self, result_root_dir: Union[str, os.PathLike], mask_max_value: int = -1
    ):
        super(WordPairReporter, self).__init__(result_root_dir, mask_max_value)
        self.reporting_methods = {
            "spearmanr": self.report_spearmanr,
            "uuas": self.report_uuas,
        }
        self.test_forbidden = {"image_examples", "write_predictions"}

    def report_spearmanr(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[list],
        split_name: str,
    ) -> Dict[str, float]:
        """Writes the Spearman correlations between predicted and true distances.

        For each word in each sentence, computes the Spearman correlation between
        all true distances between that word and all other words, and all
        predicted distances between that word and all other words.

        Computes the average such metric between all sentences of the same length.
        Writes these averages to disk.
        Then computes the average Spearman across sentence lengths 5 to 50;
        writes this average to disk.

        Args:
            predictions: A sequence of batches of predictions for a data split
            labels:
            lengths:
            observations: A sequence of batches of Observations
            split_name: the string naming the data split: {train,dev,test}
        """
        lengths_to_spearmanrs = defaultdict(list)
        for prediction_batch, label_batch, length_batch in zip(
            predictions, labels, lengths
        ):
            for idx in range(prediction_batch.size()[0]):
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                prediction = prediction_batch[idx, :length, :length].detach().cpu()
                label = label_batch[idx, :length, :length].detach().cpu().float()
                if self.mask_max_value is not None:
                    prediction[label <= self.mask_max_value] = np.nan
                    label[label <= self.mask_max_value] = np.nan
                    length = sum(label[0] > self.mask_max_value).detach().cpu().item()  # type: ignore
                spearmanrs = [
                    spearmanr(pred, gold, nan_policy="omit")
                    for pred, gold in zip(prediction, label)
                ]
                lengths_to_spearmanrs[length].extend(
                    [
                        0 if np.isnan(x.correlation) else x.correlation
                        for x in spearmanrs
                    ]
                )
        spearman_return_dict: Dict[str, float] = {}
        for length in lengths_to_spearmanrs:
            key = f"spearmanr_{length}"
            if lengths_to_spearmanrs[length]:
                spearman_return_dict[key] = float(
                    np.mean(lengths_to_spearmanrs[length])
                )
            else:
                spearman_return_dict[key] = 0.0
        if np.nan in spearman_return_dict.values():
            logger.error("!!! NAN FOUND !!!")
            pass
        with open(
            os.path.join(self.reporting_root, split_name + ".spearmanr"), "w"
        ) as f:
            for length_key in sorted(spearman_return_dict):
                f.write(
                    length_key + "\t" + str(spearman_return_dict[length_key]) + "\n"
                )

        with open(
            os.path.join(self.reporting_root, split_name + ".spearmanr-5_50-mean"), "w"
        ) as f:
            mean = np.mean(
                [
                    spearman_return_dict[f"spearmanr_{x}"]
                    for x in range(5, 51)
                    if f"spearmanr_{x}" in spearman_return_dict
                ]
            )
            f.write(str(mean) + "\n")
            spearman_return_dict["spearmanr_mean_5-50"] = float(mean)
        return spearman_return_dict

    def report_image_examples(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[Any],
        split_name: str,
    ) -> None:
        """Writes predicted and gold distance matrices to disk for the first 20
        elements of the developement set as images!

        Args:
            predictions: A sequence of batches of predictions for a data split
            labels: A sequence of labels for predictions
            lengths: A sequence of lengths
            observations: A sequence of batches of Observations
            split_name the string naming the data split: {train,dev,test}
        """
        images_printed = 0
        for prediction_batch, label_batch, length_batch, observation_batch in zip(
            predictions, labels, lengths, observations
        ):
            for idx in range(prediction_batch.size()[0]):
                length = int(length_batch[idx])
                # prediction = prediction_batch[idx, :length, :length].detach().cpu()
                label = label_batch[idx, :length, :length].detach().cpu()
                words = observation_batch[idx].sentence
                fontsize = 5 * (1 + np.sqrt(len(words)) / 200)
                plt.clf()
                ax = plt.imshow(label, cmap="hot", interpolation="nearest")
                ax.set_title("Gold Parse Distance")
                ax.set_xticks(np.arange(len(words)))
                ax.set_yticks(np.arange(len(words)))
                ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha="center")
                ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va="top")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.reporting_root, split_name + "-gold" + str(images_printed)
                    ),
                    dpi=300,
                )

                plt.clf()
                ax = plt.imshow(label, cmap="hot", interpolation="nearest")
                # ax = sns.heatmap(prediction)
                ax.set_title("Predicted Parse Distance (squared)")
                ax.set_xticks(np.arange(len(words)))
                ax.set_yticks(np.arange(len(words)))
                ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha="center")
                ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va="center")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.reporting_root, split_name + "-pred" + str(images_printed)
                    ),
                    dpi=300,
                )
                logger.debug(f"Printing {images_printed}")
                images_printed += 1
                if images_printed == 20:
                    return

    def report_uuas(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[Any],
        split_name: str,
    ) -> Dict[str, float]:
        """Computes the UUAS score for a dataset.

        From the true and predicted distances, computes a minimum spanning tree
        of each, and computes the percentage overlap between edges in all
        predicted and gold trees.

        All tokens with punctuation part-of-speech are excluded from the minimum
        spanning trees.

        Args:
            predictions: A sequence of batches of predictions for a data split
            labels: A sequence of labels for predictions
            lengths: A sequence of lengths
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
                poses = observation_batch[idx].xpos_sentence
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                assert length == len(observation_batch[idx].sentence)
                prediction = prediction_batch[idx, :length, :length].detach().cpu()
                label = label_batch[idx, :length, :length].detach().cpu()

                gold_edges = dist_matrix_to_edges(label, poses)
                pred_edges = dist_matrix_to_edges(prediction, poses)

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
