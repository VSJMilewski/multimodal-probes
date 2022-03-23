import logging
import os
import pickle
from collections import defaultdict
from typing import Dict

import h5py  # type: ignore
import numpy as np
from probing_project.utils import Observation
from rich.progress import track
from torch.nn import CrossEntropyLoss

from .edge_position_task import EdgePositionTask

logger = logging.getLogger(__name__)


class EdgePositionControlTask(EdgePositionTask):
    def __init__(self):
        super(EdgePositionControlTask, self).__init__()
        self.label_name = "control_edge_position_labels"
        self.loss = CrossEntropyLoss()
        self.control_token2edgelabel_map = defaultdict(
            lambda: int(np.random.choice([0, 1, 2]))
        )

    def add_task_label_dataset(
        self, input_conllx_pkl: str, unformatted_output_h5: str, *_, **__
    ):
        if self.control_token2edgelabel_map is None:
            self.control_token2edgelabel_map = defaultdict(
                lambda: int(np.random.choice([0, 1, 2]))
            )

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
                edge_label = f_out.create_dataset(
                    "control_edge_position_labels",
                    (num_lines, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len),
                    compression=5,
                    shuffle=True,
                )

                for index, ob in track(
                    conllx_observations.items(), description="create CONTROL EDGE"
                ):
                    sent_l = len(ob.index)
                    for i, token in enumerate(ob.sentence):
                        if self.control_token2edgelabel_map[token] == 0:
                            edge_label[index, i] = i
                        elif self.control_token2edgelabel_map[token] == 1:
                            edge_label[index, i] = 0
                        elif self.control_token2edgelabel_map[token] == 2:
                            edge_label[index, i] = i
                            edge_label[index, i] = sent_l - 1
        logger.info(f"All split files for label {self.label_name} ready!")
