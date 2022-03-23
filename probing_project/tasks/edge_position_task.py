import logging
import os
import pickle
from typing import Dict, Set

import h5py  # type: ignore
import numpy as np
from probing_project.utils import Observation
from rich.progress import track
from torch.nn import CrossEntropyLoss

from .base_task import TaskBase

logger = logging.getLogger(__name__)


class EdgePositionTask(TaskBase):
    accepted_probes: Set[str] = {
        "TwoWordNNLabelProbe",
        "TwoWordLinearLabelProbe",
        "TwoWordFeaturizedLinearLabelProbe",
    }

    def __init__(self):
        super(EdgePositionTask, self).__init__()
        self.label_name = "edge_position_labels"
        self.loss = CrossEntropyLoss()

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
                edge_label = f_out.create_dataset(
                    "edge_position_labels",
                    (num_lines, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len),
                    compression=5,
                    shuffle=True,
                )

                for index, ob in track(
                    conllx_observations.items(), description="create EDGE POS"
                ):
                    sent_l = len(ob.index)
                    edge_label[index, :sent_l] = np.array(
                        [int(x) - 1 for x in ob.head_indices]
                    )
        logger.info(f"All split files for label {self.label_name} ready!")
