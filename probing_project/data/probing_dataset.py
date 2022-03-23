import logging
import os
import pickle
import sys
from typing import List, Optional, Tuple, Union

import h5py  # type: ignore
import torch
from probing_project.tasks.base_task import TaskBase
from probing_project.utils import Observation
from torch.utils.data import Dataset

sys.path.append("../../volta")
from volta.config import BertConfig  # type: ignore

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ProbingDataset(Dataset):
    def __init__(
        self,
        h5_embeddings_file: str,
        h5_labels_file: str,
        h5_label_length_file: str,
        layer_idx: Union[int, str],
        task: TaskBase,
        sentence_file: Optional[str] = None,
        mm_config: Optional[BertConfig] = None,
        mm_bert_layer: bool = True,
        mm_vision_layer: bool = False,
    ) -> None:
        """

        Args:
            h5_embeddings_file: The path to the HDF5 file where the extracted
                                embeddings of the model are stored.
            h5_labels_file: The path to a HDF5 file where the labels are stored.
                            Each task will get a seperate file.
            layer_idx: What is the layer we are testing on.
            task: The task for which the data is used.
            sentence_file: The file path where the sentence are stored
            mm_config: when given, also set the use of multimodal to true.
            mm_bert_layer: If we should map from a BERT layer to the MM layer,
                           or use the MM layer directly.
            mm_vision_layer: If true, use vision layers instead of the language layers.
        """
        self.h5_embeddings_file = h5_embeddings_file
        self.h5_labels_file = h5_labels_file
        self.h5_label_length_file = h5_label_length_file
        self.layer_idx: str = str(layer_idx)
        self.task = task
        self.mm_vision_layer = mm_vision_layer

        self.observations: Optional[List[Observation]] = None
        if sentence_file is not None:
            with open(sentence_file, "rb") as f_sent:
                self.observations = pickle.load(f_sent)

        # setup stuff for with a multimodal bert
        if mm_config is not None:
            if mm_bert_layer:
                if mm_vision_layer:
                    layer_idx = mm_config.bert_layer2ff_sublayer[str(layer_idx)]
                    if layer_idx not in mm_config.v_ff_sublayers:
                        logger.error(
                            f"given mapped layer_idx {layer_idx} "
                            f"not a vision-feedforward sublayer"
                        )
                        raise ValueError(f"given layer_idx {layer_idx} not found")
                    self.layer_idx = "vision" + str(layer_idx)
                else:
                    self.layer_idx = "text" + str(
                        mm_config.bert_layer2ff_sublayer[str(layer_idx)]
                    )
            else:
                if mm_vision_layer:
                    if layer_idx not in mm_config.v_ff_sublayers:
                        logger.error(
                            f"given layer_idx {layer_idx} not a "
                            f"vision-feedforward sublayers"
                        )
                        raise ValueError(f"given layer_idx {layer_idx} not found")
                    self.layer_idx = "vision" + str(layer_idx)
                else:
                    if layer_idx not in mm_config.t_ff_sublayers:
                        logger.error(
                            f"given layer_idx {layer_idx} not a "
                            f"text-feedforward sublayers"
                        )
                        raise ValueError(f"given layer_idx {layer_idx} not found")
                    self.layer_idx = "text" + str(layer_idx)

        # setup the dataset size
        with h5py.File(self.h5_embeddings_file, "r", libver="latest") as f:
            self.dataset_size = len(f[self.layer_idx])

        # placeholder open h5 files
        self.open_emb_file: Union[h5py.File, None] = None
        self.open_lab_file: Union[h5py.File, None] = None
        self.open_lab_length_file: Union[h5py.File, None] = None
        self.embedding_features: Union[h5py.Dataset, None] = None
        self.labels: Union[h5py.Dataset, None] = None
        self.lengths: Union[h5py.Dataset, None] = None

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(
        self, index: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, Optional[Observation]]]:
        """
        return a single sample from the dataset
        Args:
            index: the identifier of the sample to return
        Returns:
            embbedings (Torch.Tensor): tensor with the embeddings for the sentence
            labels (torch.Tensor): tensor with labels for the current task
            length (int): The length of the labels
            observation (Observation): the complete data for the sample in connlx format

            Or None: when the lenght is 0, so  this sample can be skipped
        """
        if self.open_emb_file is None:
            # logger.debug("OPENING ANOTHER HDF5 INSTANCE")
            self.open_emb_file = h5py.File(
                self.h5_embeddings_file, "r", libver="latest"
            )
            self.embedding_features = self.open_emb_file[self.layer_idx]

        if self.open_lab_length_file is None:
            self.open_lab_length_file = h5py.File(
                self.h5_label_length_file, "r", libver="latest"
            )
            if self.mm_vision_layer:
                self.lengths = self.open_emb_file["num_boxes"]
            else:
                self.lengths = self.open_lab_length_file["label_length"]

        if self.open_lab_file is None:
            self.open_lab_file = h5py.File(self.h5_labels_file, "r", libver="latest")
            self.labels = self.open_lab_file[self.task.label_name]

        embeddings = torch.tensor(self.embedding_features[index])  # type: ignore
        labels = torch.tensor(self.labels[index], dtype=torch.long)  # type: ignore
        length: int = self.lengths[index]  # type: ignore

        if self.observations is not None:
            observation = self.observations[index]
        else:
            observation = None

        if length == 0:
            return None

        return embeddings, labels, length, observation

    def __del__(self):
        if self.open_emb_file is not None:
            self.open_emb_file.close()
        if self.open_lab_file is not None:
            self.open_lab_file.close()
