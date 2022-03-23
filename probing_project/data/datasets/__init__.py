import logging
from typing import Dict, Set, Type

from .base_dataset import DatasetBase, MultimodalDatasetBase, TextDatasetBase
from .flickr30k_dataset import Flickr30k
from .ptb3_dataset import PTB3

logger = logging.getLogger(__name__)


textclasses: Dict[str, Type[TextDatasetBase]] = {
    cls.__name__: cls for cls in TextDatasetBase.__subclasses__()
}
multimodalclasses: Dict[str, Type[MultimodalDatasetBase]] = {
    cls.__name__: cls for cls in MultimodalDatasetBase.__subclasses__()
}


def get_possible_text_datasets() -> Set[str]:
    return set(textclasses.keys())


def get_possible_multimodal_datasets() -> Set[str]:
    return set(multimodalclasses.keys())


def get_dataset_class(name: str):
    if name in textclasses:
        return textclasses[name]
    elif name in multimodalclasses:
        return multimodalclasses[name]
    else:
        logger.error("Unknown dataset class: {}".format(name))
        raise ValueError("Unknown dataset class: {}".format(name))
