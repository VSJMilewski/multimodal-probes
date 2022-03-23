import argparse
import logging
from typing import List, Set, Type, Union

import h5py  # type: ignore
from probing_project.probes import get_possible_probes
from probing_project.reporters import ReporterBase
from torch import nn

logger = logging.getLogger(__name__)


class TaskBase(object):

    accepted_probes: Set[str] = get_possible_probes()

    def __init__(self):
        """
        The base class for a task. It defines all the variables
        that are expected to be defined in every task:
            - self.label_name: the name for the label associated with the task
            - self.loss: the loss function used in the task
            - self.reporter_class: the class of the reporter
                                   that computes results for this task
        """
        self.label_name = None
        self.loss: Union[nn.Module] = nn.Module()
        self.reporter_class: Type[ReporterBase] = ReporterBase

    def get_reporter(self, result_root_dir, mask_max_value) -> ReporterBase:
        assert self.reporter_class is not None
        return self.reporter_class(result_root_dir, mask_max_value)

    def check_add_label(
        self, hdf5_label_file: str, label_name: str = None
    ) -> List[str]:
        """
        Checks for each split if the current label still needs to be computed a
        nd added to the hdf5 file
        Args:
            hdf5_label_file: the general file name, with place to enter the split.
            label_name: option to pass specific label name to check for different l
                        abel then in task

        Returns: list of splits still requiring processing
        """
        if label_name is None:
            label_name = self.label_name
        split_needed: List[str] = []
        for split in ["train", "dev", "test"]:
            with h5py.File(
                hdf5_label_file.format(split), "r", libver="latest"
            ) as f_out:
                if label_name in f_out:
                    logger.info(
                        f"Label {label_name} already added to hdf5 for split {split}"
                    )
                else:
                    split_needed.append(split)
        return split_needed

    def add_task_label_dataset(
        self, input_conllx_pkl: str, unformatted_output_h5: str, *_, **__
    ):
        logger.error(
            "Base task does not have any labels. Use specific tasks that create labels."
        )
        raise NotImplementedError()

    # ARGUMENT METHODS
    @classmethod
    def add_task_specific_args(
        cls, parent_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("TaskArgs")
        parser.add_argument(
            "--probe",
            required=True,
            type=str,
            choices=cls.accepted_probes,
            help="Select the probe that is usable for the current task."
            "When no task is specified, all existing probes are listed. "
            "See help again with a task selected to see probes that work with task.\n"
            "NOTE: Some probes create additional arguments.",
        )
        return parent_parser
