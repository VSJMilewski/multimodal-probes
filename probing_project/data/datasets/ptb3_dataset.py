import logging
import os
import subprocess
from typing import Tuple, Union

from .base_dataset import TextDatasetBase

logger = logging.getLogger(__name__)


class PTB3(TextDatasetBase):
    def __init__(self, raw_dataset_path=None, intermediate_path=None):
        super(PTB3, self).__init__()
        self.split_subsets = {
            "train": (2, 21),
            "dev": 22,
            "test": 23,
        }
        self.filename = "ptb3-wsj"
        self.raw_dataset_path = raw_dataset_path
        self.intermediate_path = intermediate_path

    def prepare_intermediate_data(self, unformat_conllx_file):
        """
        When embeddings data must me prepared, we first need to precompute the raw data
        Returns:
        """
        test_files = [
            unformat_conllx_file.format(split) for split in ["train", "dev", "test"]
        ]
        if not (
            os.path.isfile(test_files[0])
            and os.path.isfile(test_files[1])
            and os.path.isfile(test_files[2])
        ):
            logger.debug("creating intermediate files")
            self._convert_ptb3_splits_to_depparse_and_conllx(
                self.raw_dataset_path, self.intermediate_path, unformat_conllx_file
            )
            logger.info("Intermediate files ready!")
        else:
            logger.debug("all needed intermediate files already exist")

    def _convert_ptb3_splits_to_depparse_and_conllx(
        self, ptb_path: str, output_tree_path: str, output_conllx_file: str
    ):
        """
        Create the splits for penn treebank-3, using the conventional subsets.
        Train: 2-21, dev: 22, test: 23. Then the splits are converted to conllx format
        into conllx files.
        Args:
            ptb_path: Path to the root dir of the Penn Treebank-3 dataset
            output_tree_path: Where to store the created split tree files.
            output_conllx_file: Where to store the converted conllx formated tree files.
        Returns: None
        """
        #  create train/dev/test split
        logger.info(
            "Combining the subsets from PTB-3 WallStreetJournal(WSJ) into "
            "the train/dev/test splits."
        )
        output_tree_file = os.path.join(output_tree_path, "ptb3-wsj-{}.trees")
        for split in zip(["train", "dev", "test"], [(2, 21), 22, 23]):
            tree_f = output_tree_file.format(split)
            # first we combine the subsets into the splits tree files
            PTB3._create_split_wsj_to_tree(ptb_path, tree_f, self.split_subsets[split])

            # now we convert the tree format into the conllx format
            logger.info(
                "Creating the conllx file: {}".format(
                    os.path.basename(output_conllx_file.format(split))
                )
            )
            with open(output_conllx_file.format(split), "w") as out_f:
                x = subprocess.run(
                    [
                        "java",
                        "-mx1g",
                        "edu.stanford.nlp.trees.EnglishGrammaticalStructure",
                        "-treeFile",
                        tree_f,
                        "-checkConnected",
                        "-basic",
                        "-keepPunct",
                        "-conllx",
                    ],
                    stdout=out_f,
                )
                if x.returncode != 0:
                    logger.error(
                        "Running stanford coreNLP results in error. "
                        "Perhaps it is not installed or added to classpath. "
                        "Install from here: https://stanfordnlp.github.io/CoreNLP/. "
                        "!!  Only needed when preparing the data !!"
                    )
                    raise RuntimeError

    @staticmethod
    def _create_split_wsj_to_tree(
        ptb_path: str,
        output_tree_file: str,
        minmax_subsets: Union[int, Tuple[int, int]],
    ) -> None:
        """
        Combines all files in one or more subsets of the PTB WallStreetJournal(WSJ)
        data into a single file.
        Args:
            ptb_path: Path to the root dir of the Penn Treebank-3 dataset
            output_tree_file: Where to store the created split tree file.
            minmax_subsets: which subsets to use for current split. If an int is given,
                            only a single subset is used. If tuple of 2 ints is given,
                            an inclusive range for these subsets is created.

        Returns: None
        """
        logger.info(
            f"Combining the subsets {minmax_subsets} from PTB-3 WallStreetJournal(WSJ) "
            f"into the split tree file: {output_tree_file}"
        )
        wsj_path = os.path.join(ptb_path, "treebank_3/parsed/mrg/wsj/{:0>2}/")
        # make sure that minmax_subset is a tuple
        if isinstance(minmax_subsets, int):
            minmax_subsets = (minmax_subsets, minmax_subsets)
        os.makedirs(os.path.dirname(output_tree_file), exist_ok=True)
        with open(output_tree_file, "w") as f_out:
            for i in range(minmax_subsets[0], minmax_subsets[1] + 1):
                tmp_path = wsj_path.format(i)
                for f in os.listdir(tmp_path):
                    f_ = os.path.join(tmp_path, f)
                    if os.path.isfile(f_):
                        with open(f_, "r") as f_in:
                            for line in f_in:
                                f_out.write(line)
