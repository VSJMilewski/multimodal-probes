import logging
import os
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch
from matplotlib import pyplot as plt  # type: ignore
from probing_project.utils import Observation, get_nopunct_argmin
from scipy.stats import spearmanr  # type: ignore

from .reporter_base import ReporterBase

logger = logging.getLogger(__name__)


class WordReporter(ReporterBase):
    """Reporting class for single-word (depth) tasks"""

    def __init__(
        self,
        result_root_dir: Union[str, os.PathLike],
        mask_max_value: int = None,
    ):
        super(WordReporter, self).__init__(result_root_dir, mask_max_value)
        self.reporting_methods = {
            "spearmanr": self.report_spearmanr,
            "root_acc": self.report_root_acc,
        }
        self.test_forbidden = {"image_examples", "write_predictions"}

    def report_spearmanr(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[List[Observation]],
        split_name: str,
    ) -> Dict[str, float]:
        """
        Average spearman correlation over all sentences of the same length.
        Also computes average Spearman across the sentences with lengths 5 to 50.
        """
        lengths_to_spearmanrs: Dict[int, List[float]] = defaultdict(list)
        for prediction_batch, label_batch, length_batch in zip(
            predictions, labels, lengths
        ):
            for idx in range(prediction_batch.size()[0]):
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                prediction = prediction_batch[idx, :length].detach().cpu()
                label = label_batch[idx, :length].detach().cpu().float()
                if self.mask_max_value is not None:
                    prediction[label <= self.mask_max_value] = np.nan
                    label[label <= self.mask_max_value] = np.nan
                    length = sum(label > self.mask_max_value)
                if length < 3:
                    continue
                sent_spearmanr = spearmanr(prediction, label, nan_policy="omit")
                lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)
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
            for spear_length in sorted(spearman_return_dict):
                f.write(f"{spear_length}\t{str(spearman_return_dict[spear_length])}\n")
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

    def report_root_acc(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[List[Observation]],
        split_name: str,
    ) -> Dict[str, float]:
        """
        Percentage of sentences where root token is predicted as highest in the tree
        """
        total_sents = 0
        correct_root_predictions = 0
        for prediction_batch, label_batch, length_batch, observation_batch in zip(
            predictions, labels, lengths, observations
        ):
            for idx in range(prediction_batch.shape[0]):
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                label_0_index = (
                    (label_batch[idx, :length].detach() == 0)
                    .nonzero(as_tuple=True)[0]
                    .item()
                )
                prediction = prediction_batch.data[idx, :length].detach().cpu()
                words = observation_batch[idx].sentence
                poses = observation_batch[idx].xpos_sentence
                correct_root_predictions += label_0_index == get_nopunct_argmin(
                    prediction, words, poses
                )
                total_sents += 1
        root_acc = correct_root_predictions / float(total_sents)
        with open(
            os.path.join(self.reporting_root, split_name + ".root_acc"), "w"
        ) as fout:
            fout.write(
                "\t".join(
                    [str(root_acc), str(correct_root_predictions), str(total_sents)]
                )
                + "\n"
            )
        return {"root_acc": root_acc}

    def report_image_examples(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor],
        lengths: List[torch.Tensor],
        observations: List[List[Observation]],
        split_name: str,
    ) -> None:
        """
        Writes predicted and gold depths to disk for the first
        20 elements of the developement set as images!
        """
        images_printed = 0
        for prediction_batch, label_batch, length_batch, observation_batch in zip(
            predictions, labels, lengths, observations
        ):
            for idx in range(prediction_batch.shape[0]):
                plt.clf()
                length: int = length_batch[idx].detach().cpu().item()  # type: ignore
                prediction = prediction_batch[idx, :length].detach().cpu()
                label = label_batch[idx, :length].detach().cpu()
                words = observation_batch[idx].sentence
                fontsize = 6
                cumdist = 0
                for index, (word, gold, pred) in enumerate(
                    zip(words, label, prediction)
                ):
                    plt.text(
                        cumdist * 3, gold * 2, word, fontsize=fontsize, ha="center"
                    )
                    plt.text(
                        cumdist * 3,
                        pred * 2,
                        word,
                        fontsize=fontsize,
                        color="red",
                        ha="center",
                    )
                    cumdist = cumdist + (np.square(len(word)) + 1)

                plt.ylim(0, 20)
                plt.xlim(0, cumdist * 3.5)
                plt.title(
                    "LSTM H Encoder Dependency Parse Tree Depth Prediction", fontsize=10
                )
                plt.ylabel("Tree Depth", fontsize=10)
                plt.xlabel("Linear Absolute Position", fontsize=10)
                plt.tight_layout()
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=5)
                plt.savefig(
                    os.path.join(
                        self.reporting_root, split_name + "-depth" + str(images_printed)
                    ),
                    dpi=300,
                )
                images_printed += 1
                if images_printed == 20:
                    return
