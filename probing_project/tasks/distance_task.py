import logging
import os
import pickle
from typing import Dict, Set

import h5py  # type: ignore
import numpy as np
from probing_project.losses import L1DistanceLoss
from probing_project.reporters import WordPairReporter
from probing_project.utils import Observation
from rich.progress import track

from .base_task import TaskBase
from .utils import create_head_indices

logger = logging.getLogger(__name__)


class DistanceTask(TaskBase):
    accepted_probes: Set[str] = {"TwoWordPSDProbe", "TwoWordNonPSDProbe"}

    def __init__(self):
        super(DistanceTask, self).__init__()
        self.label_name = "parse_distance_labels"
        self.loss = L1DistanceLoss()
        self.reporter_class = WordPairReporter

    def add_task_label_dataset(
        self, input_conllx_pkl: str, unformatted_output_h5: str, *_, **__
    ):
        for split in ["train", "dev", "test"]:
            label_h5_file = unformatted_output_h5.format(split)
            if os.path.exists(label_h5_file):
                logging.info(
                    f"File for label {self.label_name} exists for split {split}."
                )
                continue
            logger.info(f"Create label {self.label_name} file for split {split}")
            with open(input_conllx_pkl.format(split), "rb") as in_pkl:
                conllx_observations: Dict[int, Observation] = pickle.load(in_pkl)
            num_lines = len(conllx_observations)
            max_label_len = max(len(ob.index) for ob in conllx_observations.values())
            with h5py.File(label_h5_file, "w", libver="latest") as f_out:
                distance_label = f_out.create_dataset(
                    self.label_name,
                    (num_lines, max_label_len, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len, max_label_len),
                    compression=5,
                    shuffle=True,
                )
                for index, ob in track(
                    conllx_observations.items(), description="create DIST labels"
                ):
                    # compute the labels
                    sent_l = len(ob.index)
                    distance_label[
                        index, :sent_l, :sent_l
                    ] = self.compute_parse_distance_labels(ob)
        logger.info(f"All split files for label {self.label_name} ready!")

    @staticmethod
    def compute_parse_distance_labels(observation):
        """
        Computes the distances between all pairs of words; returns them as torch tensor.
        Args:
            observation: a single Observation class for a sentence:
        Returns:
            A torch tensor of shape (sentence_length, sentence_length) of distances
            in the parse tree as specified by the observation annotation.
        """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        distances = np.zeros((sentence_length, sentence_length))
        head_indices = create_head_indices(observation)
        for i in range(sentence_length):
            for j in range(i, sentence_length):
                i_j_distance = DistanceTask._distance_between_pairs(i, j, head_indices)
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance
        return distances

    @staticmethod
    def _distance_between_pairs(i, j, head_indices):
        """Computes path distance between a pair of words
        Args:
          i: one of the two words to compute the distance between.
          j: one of the two words to compute the distance between.
          head_indices: the head indices (according to a dependency parse) of all words.
        Returns:
          The integer distance d_path(i,j)
        """
        if i == j:
            return 0
        i_path = [i + 1]
        j_path = [j + 1]
        i_head = i + 1
        j_head = j + 1
        while True:
            if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
                i_head = head_indices[i_head - 1]
                i_path.append(i_head)
            if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
                j_head = head_indices[j_head - 1]
                j_path.append(j_head)
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break
        total_length = j_path_length + i_path_length
        return total_length
