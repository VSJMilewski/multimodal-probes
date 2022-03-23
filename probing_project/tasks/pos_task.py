import logging
import os
import pickle
from collections import Counter
from typing import Dict, Set, Union

import h5py  # type: ignore
import numpy as np
from probing_project.utils import Observation
from rich.progress import track
from torch.nn import CrossEntropyLoss

from .base_task import TaskBase

logger = logging.getLogger(__name__)


class POSTask(TaskBase):
    accepted_probes: Set[str] = {"OneWordLinearLabelProbe", "OneWordNNLabelProbe"}

    def __init__(self):
        super(POSTask, self).__init__()
        self.label_name = "pos_tag_labels"
        self.loss = CrossEntropyLoss()
        self.vocab_file = "pos_tag_vocab.pkl"

    @staticmethod
    def create_vocab(input_conllx_pkl: str, vocab_file: Union[str, os.PathLike]):
        """
        Args:
            input_conllx_pkl: Path to the conllx pickle,
                              unformatted so split can be added
            vocab_file: Where to store the created vocab pickle.
        Returns: None
        """
        logger.info("Creating the POS tag vocabulary")
        if os.path.isfile(vocab_file):
            logger.info("POS tag vocabulary already exists.")
            return
        # creating the vocab
        vocab: Dict[str, int] = {}
        voc_count: Counter = Counter()
        # considering all possible tags in all splits
        for split in ["train", "dev", "test"]:
            with open(input_conllx_pkl.format(split), "rb") as in_pkl:
                conllx_observations: dict = pickle.load(in_pkl)
                for ob in conllx_observations.values():
                    for pos_tag in ob.xpos_sentence:
                        if pos_tag not in vocab:
                            vocab[pos_tag] = len(vocab)
                        voc_count[pos_tag] += 1
        total = np.sum(list(voc_count.values()))
        voc_dist = {tag: cnt / total for tag, cnt in voc_count.items()}
        with open(vocab_file, "wb") as f_out:
            pickle.dump({"vocabulary": vocab, "tag_distribution": voc_dist}, f_out)
        logger.info("Created the POS tag vocabulary with {} tokens".format(len(vocab)))

    def add_task_label_dataset(
        self, input_conllx_pkl: str, unformatted_output_h5: str, *_, **__
    ):
        if not os.path.exists(self.vocab_file):
            self.create_vocab(input_conllx_pkl, self.vocab_file)
        with open(self.vocab_file, "rb") as f_voc:
            vocab = pickle.load(f_voc)
        voc = vocab["vocabulary"]

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
                    conllx_observations.items(), description="create POS labels"
                ):
                    sent_l = len(ob.index)
                    for sent_i in range(sent_l):
                        pos_label[index, sent_i] = voc[ob.xpos_sentence[sent_i]]
        logger.info(f"All split files for label {self.label_name} ready!")
