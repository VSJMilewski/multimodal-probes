import logging

logger = logging.getLogger(__name__)


class DatasetBase:
    def __init__(self):
        pass


class TextDatasetBase(DatasetBase):
    def __init__(self):
        super(TextDatasetBase, self).__init__()
        pass


class MultimodalDatasetBase(DatasetBase):
    def __init__(self):
        super(MultimodalDatasetBase, self).__init__()
        pass
