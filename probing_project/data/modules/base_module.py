import argparse
import logging
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Union

import h5py  # type: ignore
import pytorch_lightning as pl
from probing_project import constants
from probing_project.data.datasets import get_dataset_class
from probing_project.data.utils import filter_none_from_updated_default_collate
from probing_project.tasks import TaskBase
from probing_project.utils import (
    Observation,
    progress_columns,
    str2bool,
    track,
)
from rich.progress import Progress
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ProbingModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: Union[str, os.PathLike],
        bert_model: str,
        bert_cased: bool,
        embeddings_model: str,
        bert_model_layer: str,
        dataset: str,
        num_workers: int,
        batch_size: int,
        task: TaskBase,
        *_,
        **__,  # important, since the entire argparse dict is given. But we never use them
    ) -> None:
        """
        Args:
            data_root_dir: The root directory where all data is stored
            bert_model: What is the initial bert model: 'base' or 'large'
            bert_cased: use cased 'True' or uncased 'False' bert.
            bert_model_layer: Which layer are we using to train our probe on.
            embeddings_model: Which embeddings model to use: 'BERT', 'ViLBERT', ...
            dataset: Which dataset to use, options in DATASETS contsant
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            task: one of TASK_CLASSES. Select labels to use and set-up task settings
        """
        super().__init__(
            train_transforms=None, val_transforms=None, test_transforms=None, dims=None
        )
        # setup bert model naming and stuff
        case_str = "cased" if bert_cased else "uncased"
        self.model_name = f"BERT-{bert_model}-{case_str}"
        self.bert_model = bert_model
        self.bert_cased = bert_cased
        self.embeddings_model = embeddings_model
        self.bert_model_layer = bert_model_layer
        self.dataset_name = dataset
        self.task = task

        # create initial needed paths and directories
        self._setup_dirs(data_root_dir)

        # setup dataset stuff
        dataset_class = get_dataset_class(dataset)
        self.dataset = dataset_class(
            raw_dataset_path=self.dataset_path,
            intermediate_path=self.general_intermediate_path,
        )

        # create remaining paths, now we know the dataset
        self._setup_data_specific_dirs_and_dirs()

        # setup current experiment settings
        self.num_workers = num_workers
        self.batch_size = batch_size

        # will be defined in setup(). make sure to run prepare_data() first.
        self.train_dataset: Union[Dataset, None] = None
        self.val_dataset: Union[Dataset, None] = None
        self.test_dataset: Union[Dataset, None] = None

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Checks if all necesary data is created and downloaded.
        If something is missing, it will create it. If it is some file that cannot
        be created by the datamodule, it will provide instructions on what to do.
        """
        super().prepare_data(*args, **kwargs)

        logger.info("Preparing Data")
        # The conllx pickles are required training and creating the h5s.
        # For this, we need the intermediate data.
        self.dataset.prepare_intermediate_data(self.unformatted_input_conllx)
        self._prepare_conllx_dict_pickle()

        # CHECK FOR EMBEDDING FEATURE HDF5 and create if necessary
        self._check_and_create_embedding_features_file()

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("Use child classes")

    def train_dataloader(self) -> DataLoader:
        """
        Creates the dataloader with the training data and returns it.
        Returns: A pytorch dataloader for the training dataset
        """
        logger.debug("Creating train loader")
        assert self.train_dataset is not None, "make sure to run 'setup()' first"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=filter_none_from_updated_default_collate,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates the dataloader with the validation data and returns it.
        Returns: A pytorch dataloader for the validation dataset
        """
        logger.debug("Creating val loader")
        assert self.val_dataset is not None, "make sure to run 'setup()' first"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=filter_none_from_updated_default_collate,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates the dataloader with the testing data and returns it.
        Returns: A pytorch dataloader for the testing dataset
        """
        logger.debug("Creating test loader")
        assert self.test_dataset is not None, "make sure to run 'setup()' first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=filter_none_from_updated_default_collate,
        )

    def _setup_dirs(self, data_dir: Optional[Union[str, os.PathLike]]):
        # setup the main data directories
        self.data_dir = (
            os.path.join(os.getcwd(), "..") if data_dir is None else data_dir
        )
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.int_dir = os.path.join(self.data_dir, "intermediate")
        self.pro_dir = os.path.join(self.data_dir, "processed")
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.isdir(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.isdir(self.int_dir):
            os.makedirs(self.int_dir)
        if not os.path.isdir(self.pro_dir):
            os.makedirs(self.pro_dir)

        self.dataset_path = os.path.join(self.raw_dir, self.dataset_name)
        self.general_processed_path = os.path.join(self.pro_dir, self.dataset_name)
        self.general_intermediate_path = os.path.join(self.int_dir, self.dataset_name)
        os.makedirs(self.general_processed_path, exist_ok=True)
        os.makedirs(self.general_intermediate_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)

    def _setup_data_specific_dirs_and_dirs(self):
        # create paths to specific files
        self.unformatted_input_conllx = os.path.join(
            self.general_intermediate_path, self.dataset.filename + "-{}.conllx"
        )
        self.unformatted_conllx_pickle = os.path.join(
            self.general_processed_path, self.dataset.filename + "-{}.conllx.pkl"
        )

        # create paths for storing embeddings
        embedding_hdf5_path = os.path.join(
            self.general_processed_path, self.model_name, self.embeddings_model
        )
        self.unformatted_emb_h5_file = os.path.join(
            embedding_hdf5_path, self.dataset.filename + "-{}.h5"
        )
        label_hdf5_path = os.path.join(self.general_processed_path, "labels")
        self.unformatted_label_h5_file = os.path.join(
            label_hdf5_path,
            self.dataset.filename + "-{}." + self.task.label_name + ".h5",
        )
        self.unformatted_length_label_h5_file = os.path.join(
            label_hdf5_path, self.dataset.filename + "-{}.label_length.h5"
        )
        os.makedirs(embedding_hdf5_path, exist_ok=True)
        os.makedirs(label_hdf5_path, exist_ok=True)

    def _prepare_conllx_dict_pickle(self) -> None:
        for split in ["train", "dev", "test"]:
            input_conllx = self.unformatted_input_conllx.format(split)
            conllx_pickle = self.unformatted_conllx_pickle.format(split)
            if not os.path.isfile(conllx_pickle):
                logger.info(f"Creating conllx pickle for split {split}")
                conllx_observations = self._load_conllx_dataset(input_conllx)
                conllx_dict = {}
                logger.info(
                    f"preparing the conllx file into a pickle of Observations "
                    f"for split {split}"
                )
                for index, ob in track(
                    enumerate(conllx_observations),
                    total=len(conllx_observations),
                    description="2conllx_pickle",
                ):
                    conllx_dict[index] = ob
                with open(os.path.join(conllx_pickle), "wb") as sent_f:
                    pickle.dump(conllx_dict, sent_f)
            else:
                logger.info(f"Conllx pickle for split {split} already exists")
        logger.info("Conllx pickles for all splits ready!")

    def _load_conllx_dataset(self, filepath: str) -> List[Observation]:
        """Reads in a conllx file; generates Observation objects
        For each sentence in a conllx file, generates a single Observation object.
        Args:
            filepath: the filesystem path to the conllx dataset
        Returns: A list of Observations namedtuples
        """
        observations = []
        lines = list(x for x in open(filepath))
        # get list of conllx describtions. yields a single list per sentence
        logger.info(f"Loading conllx into observations: {os.path.basename(filepath)}")
        with Progress(*progress_columns) as prog:
            t = prog.add_task(
                description="split conllx into sents pickle", total=len(lines)
            )
            for buf in self._generate_lines_for_sent(lines):
                conllx_lines = []
                for line in buf:
                    conllx_lines.append(line.strip().split("\t"))
                observation = Observation(*zip(*conllx_lines))
                observations.append(observation)
                prog.advance(t, len(buf))
            return observations

    def _check_and_create_embedding_features_file(self):
        logger.error("Please use a subclass to create embedding features")
        raise NotImplementedError

    @staticmethod
    def _generate_lines_for_sent(lines: List):
        """Yields batches of lines describing a sentence in conllx.
        Args:
            lines: Each line of a conllx file.
        Yields: a list of lines describing a single sentence in conllx.
        """
        buf: List[str] = []
        for line in lines:
            if line.startswith("#"):
                continue
            if not line.strip():  # it is an empty line, so end of sentence
                if buf:
                    yield buf
                    buf = []
                else:  # nothing in buffer, so nothing to return
                    continue
            else:
                buf.append(line.strip())
        if buf:
            yield buf

    # LABELS COMPUTE
    def _create_label_h5(
        self, task: TaskBase, input_conllx_pkl: str, output_file: str, *_, **__
    ) -> None:
        self._add_label_length_dataset()

    def _add_label_length_dataset(self):
        label_name = "label_length"
        for split in ["train", "dev", "test"]:
            leng_label_h5_file = self.unformatted_length_label_h5_file.format(split)
            if os.path.exists(leng_label_h5_file):
                logging.info(
                    f"File for label {label_name} already exists for split {split}."
                )
            else:
                logger.info(f"Create label {label_name} file for split {split}")
                with open(self.unformatted_conllx_pickle.format(split), "rb") as in_pkl:
                    conllx_observations: Dict[int:Observation] = pickle.load(in_pkl)
                num_lines = len(conllx_observations)
                with h5py.File(leng_label_h5_file, "w", libver="latest") as f_out:
                    label_len = f_out.create_dataset(
                        label_name, (num_lines,), dtype="int"
                    )
                    for index, ob in track(
                        conllx_observations.items(),
                        total=len(conllx_observations),
                        description="create LENGTH labels",
                    ):
                        # compute the labels
                        sent_l = len(ob.index)
                        label_len[index] = sent_l
        logger.info(f"All split files for label {label_name} ready!")

    # SUPPORT METHODS
    @staticmethod
    def get_bert_info(bert_model):
        if bert_model == "base":
            layer_count = 12
            feature_count = 768
        elif bert_model == "large":
            layer_count = 24
            feature_count = 1024
        else:
            logger.error("BERT model must be base or large")
            raise ValueError
        return layer_count, feature_count

    @staticmethod
    def _match_tokenized_to_untokenized(
        tokenized_sent: List[str], untokenized_sent: List[str]
    ) -> Dict:
        """Aligns tokenized and untokenized sentence given subwords "##" prefixed

        Assuming that each subword token that does not start a new word is prefixed
        by two hashes, "##", computes an alignment between the un-subword-tokenized
        and subword-tokenized sentences.

        Args:
            tokenized_sent: a list of strings describing a subword-tokenized sentence
            untokenized_sent: a list of strings describing a sentence, no subword tok.

        Returns: A dictionary of type {int: list(int)} mapping each untokenized sentence
                 index to a list of subword-tokenized sentence indices
        """
        mapping = defaultdict(list)
        untokenized_sent_index = 0
        tokenized_sent_index = 1
        # while we haven't reached the last token in untokenized or tokenized
        while untokenized_sent_index < len(
            untokenized_sent
        ) and tokenized_sent_index < len(tokenized_sent):
            # find all the sub tokens of the current word.
            # So each untokenized index can have a list of multiple tokenized indices
            while tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[
                tokenized_sent_index + 1
            ].startswith("##"):
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                tokenized_sent_index += 1
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            untokenized_sent_index += 1
            tokenized_sent_index += 1
        return mapping

    # ARGUMENT METHODS
    @staticmethod
    def add_datamodule_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("ModuleArgs")
        parser.add_argument(
            "--data_root_dir",
            default="./data/",
            type=str,
            help="The base of the path where all data is stored",
        )
        parser.add_argument(
            "--bert_model",
            default="base",
            type=str,
            choices=constants.BERT_MODELS,
            help="The possible variants of bert, I.E. 'base' and 'large'",
        )
        parser.add_argument(
            "--bert_cased",
            action="store_true",
            help="Call this if you want to use case sensitive BERT.",
        )
        parser.add_argument(
            "--embeddings_model",
            required=True,
            type=str,
            choices=constants.EMBEDDING_MODELS,
            help="What is the model to extract embeddings",
        )
        parser.add_argument(
            "--bert_model_layer",
            default="1",
            type=str,
            help="The layer of the model from which we use embeddings. "
            "For BERT-base 0-11 and for large 0-23.",
        )
        parser.add_argument(
            "--dataset",
            required=True,
            type=str,
            choices=constants.DATASETS,
            help="The dataset we want to train and test our probes on",
        )
        parser.add_argument(
            "--num_workers",
            default=8,
            type=int,
            help="The number of workers the dataloader uses for preparing data",
        )
        parser.add_argument(
            "--batch_size",
            default=32,
            type=int,
            help="Batch size for training the probe. ",
        )
        parser.add_argument(
            "--butd_root",
            default="/opt/butd/",
            type=str,
            help="The root of bottom-up-attention repo. "
            "Do not change if using provided 'airplays' docker file.",
        )
        parser.add_argument(
            "--only_text",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="use visual dataset and multimodal model, without the image features.",
        )
        return parent_parser

    @staticmethod
    def check_datamodule_args(args: argparse.Namespace):
        if args.embeddings_model in constants.MULTIMODAL_MODELS:
            assert (
                args.bert_model == "base" and not args.bert_cased
            ), "the volta visual linguistic berts are all based on bert-base-uncased"

        layer_count, feature_count = ProbingModuleBase.get_bert_info(args.bert_model)

        if args.bert_model_layer in ["text_baseline", "vision_mapped_baseline"]:
            args.__setattr__("model_hidden_dim", 768)
        elif args.bert_model_layer == "vision_baseline":
            args.__setattr__("model_hidden_dim", 2048)
        else:
            if int(args.bert_model_layer) < layer_count:
                args.__setattr__("model_hidden_dim", feature_count)
            else:
                logger.error(
                    f"layer {args.bert_model_layer} does not exist. "
                    f"Max for bert-{args.bert_model} is {args.layer_count - 1}"
                    f"(0-indexed). Also possible is one of "
                    f"'text_baseline', 'vision_baseline', or 'vision_mapped_baseline'"
                )
                raise ValueError
