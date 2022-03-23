import logging
from typing import Set

from probing_project.utils import get_all_subclasses

from .one_word_linear_label_probe import OneWordLinearLabelProbe
from .one_word_nn_label_probe import OneWordNNLabelProbe
from .one_word_nn_probe import OneWordNNProbe
from .one_word_non_psd_probe import OneWordNonPSDProbe
from .one_word_psd_probe import OneWordPSDProbe
from .probe_base import ProbeBase
from .two_word_non_psd_probe import TwoWordNonPSDProbe
from .two_word_psd_probe import TwoWordPSDProbe

logger = logging.getLogger(__name__)

# assumes everything inherits from TaskBase

name2subclass_map = {cls.__name__: cls for cls in get_all_subclasses(ProbeBase)}


def get_probe_class(name: str) -> ProbeBase:
    if name in name2subclass_map:
        return name2subclass_map[name]
    else:
        logger.error(f"Task class does not exist: {name}")
        raise ValueError(f"Task class does not exist: {name}")


def get_possible_probes() -> Set[str]:
    return set(name2subclass_map.keys())
