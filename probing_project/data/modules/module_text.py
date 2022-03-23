import csv
import logging
import os
import pickle
import sys
from typing import Optional, Union

import h5py  # type: ignore
import numpy as np
import torch
from probing_project.constants import TEXT_MODELS
from probing_project.data.probing_dataset import ProbingDataset
from probing_project.tasks import TaskBase
from probing_project.utils import track  # type: ignore
from pytorch_pretrained_bert import BertModel, BertTokenizer  # type: ignore

sys.path.append("../volta")
sys.path.append("volta")
from volta.config import BertConfig  # type: ignore
from volta.encoders import BertForVLPreTraining  # type: ignore

from .base_module import ProbingModuleBase

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
csv.field_size_limit(sys.maxsize)


class ProbingModuleText(ProbingModuleBase):
    def __init__(
        self,
        *args,
        **kwargs,  # important, since the entire argparse dict is given. But we never use them
    ) -> None:
        """
        Args:
            task: one of the TASK_CLASSES. Defines which labels to use and what
                  should be predicted
            data_dir: The root directory where all data is stored
            dataset: Which dataset to use, options in DATASETS literal
            embeddings_model: Which BERT embeddings model to use:
                              Basic 'bert' or a 'multimodal_bert'
            bert_model: What is the initial bert model: 'base' or 'large'
            bert_cased: use cased 'True' or uncased 'False' bert.
            bert_model_layer: Which layer are we using to train our probe on.
        """
        super().__init__(*args, **kwargs)

    # DATASET CREATION AND FILE CREATION
    def _check_and_create_embedding_features_file(self):
        """
        First determines if the given embbedding based model is impemented to be used.
        Next checks for each split if the needed hdf5 file is created
        """
        if self.embeddings_model not in TEXT_MODELS:
            logger.error(
                f"Given Embedding type for 'embeddings_model' not implemented. "
                f"for current datamodule class. Available options: {TEXT_MODELS}."
            )
            raise ValueError()
        for split in ["train", "dev", "test"]:
            # check if the saved embeddings hdf5 files exist.
            # Otherwise extract the embeddings. TAKES LONG!
            outfile = self.unformatted_emb_h5_file.format(split)
            if not os.path.isfile(outfile):
                logger.info(f"Creating embeddings h5 file for split {split}")
                self._convert_conllx_to_bert(
                    input_conllx_pkl=self.unformatted_conllx_pickle.format(split),
                    output_file=outfile,
                    model_name=self.model_name,
                    bert_cased=self.bert_cased,
                    bert_model=self.bert_model,
                )
            else:
                logger.info(
                    f"embeddings file '{os.path.basename(outfile)}' already exist."
                )
        logger.info("All embedding h5 files ready!")

    @staticmethod
    def _convert_conllx_to_bert(
        input_conllx_pkl: str,
        output_file: str,
        model_name: str,
        bert_cased: bool,
        bert_model: str,
        *_,
        **__,
    ) -> None:
        """
        Converts the raw file into a HDF5 file with embeddings create by
        BERT and BERT-Tokenizer. Crucially, do not do basic tokenization;
        PTB is tokenized. Just do wordpiece tokenization.
        Args:
            input_conllx_pkl: The filepath to the conllx file
            output_file: the filepath for where to store the hdf5 features
            model_name: the complete model_name as defined by 'pretrained_transformers'
            bert_cased: If the model should use cased (True) tokens or uncased
                        (False) tokens, default = False
            bert_model: whether to use BERT 'base' or 'large'
        Returns: None
        """
        logger.info(
            f"from: {os.path.basename(input_conllx_pkl)}, "
            f"creating hdf5: {'/'.join(output_file.split('/')[-2:])}"
        )
        logger.info("making use of the bert model: {model_name.lower()}")
        tokenizer = BertTokenizer.from_pretrained(
            model_name.lower(),
            do_lower_case=not bert_cased,
            cache_dir="../pytorch_pretrained_bert/",
        )
        model = BertModel.from_pretrained(
            model_name.lower(), cache_dir="../pytorch_pretrained_bert/"
        )

        def forward_hook(module, input_, output):
            nonlocal baseline_embs
            baseline_embs = output

        _ = model.embeddings.register_forward_hook(forward_hook)

        layer_count, feature_count = ProbingModuleText.get_bert_info(bert_model)
        model.eval()

        # find some statistics for creating the h5 datasets
        with open(input_conllx_pkl, "rb") as in_pkl:
            conllx_observations: dict = pickle.load(in_pkl)
        num_lines = len(conllx_observations)
        max_label_len = max(len(ob.index) for ob in conllx_observations.values())

        # start creating the hdf5
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w", libver="latest") as f_out:
            layer_sets = []
            for layer in range(layer_count):
                layer_sets.append(
                    f_out.create_dataset(
                        name=str(layer),
                        shape=(num_lines, max_label_len, feature_count),
                        fillvalue=0,
                        chunks=(1, max_label_len, feature_count),
                        compression=9,
                        shuffle=True,
                    )
                )
            text_baseline = f_out.create_dataset(
                name="text_baseline",
                shape=(num_lines, max_label_len, feature_count),
                fillvalue=0,
                dtype="f",
                chunks=(1, max_label_len, feature_count),
                compression=9,
                shuffle=True,
            )

            # start filling the hdf5
            for index, ob in track(conllx_observations.items(), description="raw2bert"):
                baseline_embs: Union[torch.Tensor, None] = None
                line = "[CLS] " + " ".join(ob.sentence) + " [SEP]"
                tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segment_ids = [1 for _ in tokenized_text]

                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])
                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor, segments_tensors)
                # place output into the datasets per layer
                untokenized_sent = ob.sentence
                untok_tok_mapping = ProbingModuleText._match_tokenized_to_untokenized(
                    tokenized_text, untokenized_sent
                )
                for layer in range(layer_count):
                    single_layer_features = (
                        encoded_layers[layer].squeeze(0).detach().cpu().numpy()
                    )
                    # WE WANT TO PROBE FOR THE TREE FOR COMPLETE WORDS.
                    # THERFORE WE FIND HOW MANY SUBTOKENS MAP TO COMPLETE WORDS.
                    # NOW WE TAKE THE AVERAGE OF THE EMBEDDINGS FOR THE SUBTOKENS,
                    # TO GET A SINGLE EMBEDDING FOR EVERY WORD.
                    for token_idx in range(len(untokenized_sent)):
                        layer_sets[layer][index, token_idx, :] = np.mean(
                            single_layer_features[
                                untok_tok_mapping[token_idx][0] : untok_tok_mapping[
                                    token_idx
                                ][-1]
                                + 1,
                                :,
                            ],
                            axis=0,
                        )
                # also store the raw baseline embeddings
                assert baseline_embs is not None
                baseline_embs = baseline_embs.squeeze(0).detach().cpu().numpy()
                for token_idx in range(len(untokenized_sent)):
                    text_baseline[index, token_idx, :] = np.mean(
                        baseline_embs[  # type: ignore
                            untok_tok_mapping[token_idx][0] : untok_tok_mapping[
                                token_idx
                            ][-1]
                            + 1,
                            :,
                        ],
                        axis=0,
                    )

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Checks if all necesary data is created and downloaded.
        If something is missing, it will create it. If it is some file that cannot
        be created by the datamodule, it will provide instructions on what to do.
        """
        super().prepare_data(*args, **kwargs)

        # CHECK FOR LABEL HDF5
        self._create_label_h5(
            task=self.task,
            input_conllx_pkl=self.unformatted_conllx_pickle,
            output_file=self.unformatted_label_h5_file,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setups the datamodule with datasets for the different splits
        Args:
            stage: If 'None', load train, val, and test split.
                   If 'fit', load only the train and val split.
                   if 'test', load only the test split.
        Returns: None
        """
        logger.debug("Setting up Data")
        assert (
            self.task is not None
        ), "Preparing can be done without task, for setup it must be set"
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ProbingDataset(
                h5_embeddings_file=self.unformatted_emb_h5_file.format("train"),
                h5_labels_file=self.unformatted_label_h5_file.format("train"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "train"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
            )
            self.val_dataset = ProbingDataset(
                h5_embeddings_file=self.unformatted_emb_h5_file.format("dev"),
                h5_labels_file=self.unformatted_label_h5_file.format("dev"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "dev"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
                sentence_file=self.unformatted_conllx_pickle.format("dev"),
            )
        if stage == "test" or stage is None:
            self.test_dataset = ProbingDataset(
                h5_embeddings_file=os.path.join(
                    self.unformatted_emb_h5_file.format("test")
                ),
                h5_labels_file=self.unformatted_label_h5_file.format("test"),
                h5_label_length_file=self.unformatted_length_label_h5_file.format(
                    "test"
                ),
                mm_vision_layer="Visual" in self.task.__class__.__name__,
                layer_idx=self.bert_model_layer,
                task=self.task,
                sentence_file=self.unformatted_conllx_pickle.format("test"),
            )

    # LABELS COMPUTE
    def _create_label_h5(
        self, task: TaskBase, input_conllx_pkl: str, output_file: str, *_, **__
    ) -> None:
        super()._create_label_h5(task, input_conllx_pkl, output_file)
        self.task.add_task_label_dataset(input_conllx_pkl, output_file)
