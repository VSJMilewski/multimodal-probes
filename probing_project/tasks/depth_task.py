import logging
import os
import pickle
from typing import Dict, List, Set

import h5py  # type: ignore
import numpy as np
from probing_project.losses import L1DepthLoss
from probing_project.reporters import WordReporter
from probing_project.utils import Observation
from rich.progress import track

from .base_task import TaskBase
from .utils import create_head_indices

logger = logging.getLogger(__name__)


class DepthTask(TaskBase):
    accepted_probes: Set[str] = {
        "OneWordPSDProbe",
        "OneWordNonPSDProbe",
        "OneWordNNProbe",
    }

    def __init__(self):
        super(DepthTask, self).__init__()
        self.label_name = "parse_depth_labels"
        self.loss = L1DepthLoss()
        self.reporter_class = WordReporter

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
                depth_label = f_out.create_dataset(
                    self.label_name,
                    shape=(num_lines, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len),
                    compression=5,
                    shuffle=True,
                )
                for index, ob in track(
                    conllx_observations.items(), description="create DEPTH labels"
                ):
                    # compute the labels
                    sent_l = len(ob.index)
                    depth_label[index, :sent_l] = self.compute_parse_depth_labels(ob)
        logger.info("All split files for label {} ready!".format(self.label_name))

    @staticmethod
    def compute_parse_depth_labels(observation: Observation) -> np.ndarray:
        """
        Computes the depth of each word; returns them as a torch tensor.
        Args:
          observation: a single Observation class for a sentence:
        Returns:
          A torch tensor of shape (sentence_length,) of depths
          in the parse tree as specified by the observation annotation.
        """
        sentence_length = len(
            observation[0]
        )  # All observation fields must be of same length
        depths = np.zeros(sentence_length)
        head_indices = create_head_indices(observation)
        for i in range(sentence_length):
            depths[i] = DepthTask.get_ordering_index(i, head_indices)
        return depths

    @staticmethod
    def get_ordering_index(i: int, head_indices: List[int]):
        """
        Computes tree depth for a single word in a sentence
        Args:
            i: the word in the sentence to compute the depth of
            head_indices: the head indices (according to a dependency parse) of all words.
        Returns: The integer depth in the tree of word i
        """
        length = 0
        i_head = i + 1  # head_indices start at 1
        while True:
            i_head = head_indices[i_head - 1]
            if length > 100:
                logger.error(
                    "A highly possible infinite loop... "
                    "the found depth already over 100. terminating"
                )
                return length
            elif i_head != 0:
                length += 1
            else:
                return length
