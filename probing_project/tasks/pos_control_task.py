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

from .pos_task import POSTask

logger = logging.getLogger(__name__)


class POSControlTask(POSTask):
    def __init__(self):
        super(POSControlTask, self).__init__()
        self.label_name = "control_pos_tag_labels"
        self.loss = CrossEntropyLoss()
        self.control_token2postag_map = None

    @staticmethod
    def _create_random_vocab_map(vocab_file):
        with open(vocab_file, "rb") as f_vocab:
            pos_vocab = pickle.load(f_vocab)
        choice_keys = []
        choice_p = []
        for k, p in pos_vocab["tag_distribution"].items():
            choice_keys.append(k)
            choice_p.append(p)
        return defaultdict(
            lambda: int(np.random.choice(range(len(choice_keys)), p=choice_p))
        )

    def add_task_label_dataset(
        self, input_conllx_pkl: str, unformatted_output_h5: str, *_, **__
    ):
        if not os.path.exists(self.vocab_file):
            self.create_vocab(input_conllx_pkl, self.vocab_file)
        self.control_token2postag_map = self._create_random_vocab_map(self.vocab_file)

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
                pos_label = f_out.create_dataset(
                    self.label_name,
                    (num_lines, max_label_len),
                    fillvalue=-1,
                    dtype="int",
                    chunks=(1, max_label_len),
                    compression=5,
                    shuffle=True,
                )

                for index, ob in track(
                    conllx_observations.items(), description="create CONTROL POS"
                ):
                    sent_l = len(ob.index)
                    for sent_i in range(sent_l):
                        token = ob.sentence[sent_i]
                        pos_label[index, sent_i] = self.control_token2postag_map[token]
        logger.info(f"All split files for label {self.label_name} ready!")
