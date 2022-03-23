import json
import logging
import os
from typing import Callable, Dict, Optional, Set, Union

logger = logging.getLogger(__name__)


class ReporterBase:
    """Base class for reporting."""

    def __init__(
        self,
        result_root_dir: Union[str, os.PathLike],
        mask_max_value: Optional[int] = None,
    ):
        self.reporting_root: Union[str, os.PathLike] = result_root_dir
        self.mask_max_value: Optional[int] = mask_max_value
        self.reporting_methods: Dict[str, Callable] = {}
        self.test_forbidden: Optional[Set[str]] = None

    def __call__(self, predictions, labels, lengths, observations, split_name):
        """
        Performs all reporting methods as specifed in the yaml experiment config dict.
        Args:
            predictions: A sequence of batches of predictions for a data split
            labels:
            lengths:
            observations:
            split_name: the string naming the data split: {train,dev,test}
        """
        return_dict = {}
        for method_name, method in self.reporting_methods.items():
            if split_name == "test" and method_name in self.test_forbidden:
                logger.debug(
                    "Reporting method {} not in test set "
                    "reporting methods (reporter.py); skipping".format(method)
                )
                continue
            logger.debug("Reporting {} on split {}".format(method_name, split_name))
            method_dict = method(predictions, labels, lengths, observations, split_name)
            if method_dict is not None:
                return_dict.update(method_dict)
        return return_dict

    def write_json(self, prediction_batches, dataset, split_name):
        """Writes observations and predictions to disk.

        Args:
            prediction_batches: A sequence of batches of predictions for a data split
            dataset: A sequence of batches of Observations
            split_name: the string naming the data split: {train,dev,test}
        """
        json.dump(
            [prediction_batch.tolist() for prediction_batch in prediction_batches],
            open(os.path.join(self.reporting_root, split_name + ".predictions"), "w"),
        )
        json.dump(
            [
                [x[0][:-1] for x in observation_batch]
                for _, _, _, observation_batch in dataset
            ],
            open(os.path.join(self.reporting_root, split_name + ".observations"), "w"),
        )
        return
