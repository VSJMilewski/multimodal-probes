import argparse
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class ProbeBase(nn.Module):
    pass

    @staticmethod
    def add_probe_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        _ = parent_parser.add_argument_group("ProbeArgs")
        return parent_parser
