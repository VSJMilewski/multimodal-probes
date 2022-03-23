import logging

from probing_project.constants import MULTIMODAL_DATASETS, TEXT_DATASETS
from probing_project.data.datasets import DatasetBase

from .base_module import ProbingModuleBase
from .module_multimodal import ProbingModuleMultimodal
from .module_text import ProbingModuleText

logger = logging.getLogger(__name__)


def get_module_class(dataset_name: DatasetBase):
    if dataset_name in TEXT_DATASETS:
        return ProbingModuleText
    elif dataset_name in MULTIMODAL_DATASETS:
        return ProbingModuleMultimodal
    else:
        logger.error(
            f"dataset not found in constants, cannot load module: {dataset_name}"
        )
        raise ValueError(
            f"dataset not found in constants, cannot load module: {dataset_name}"
        )
