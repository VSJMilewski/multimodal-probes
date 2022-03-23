import logging
import os
from typing import List, Tuple, Union

import pytorch_lightning as pl
from probing_project.constants import EMBEDDING_MAPPINGS
from probing_project.embedding_mappings import get_mapping_class
from probing_project.probes import get_probe_class
from probing_project.tasks import TaskBase
from probing_project.utils import str2bool
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


class ProbingModel(pl.LightningModule):
    def __init__(
        self,
        task: TaskBase,
        probe_group_args,
        results_root_dir: Union[str, bytes, os.PathLike],
        embedding_process_model: str,
        probe: str,
        model_hidden_dim: int,
        use_only_caption_regions: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # the maximum value that should be masked out.
        # text is -2, if only regions that are in the caption are used, we use -2.
        # otherwise, use -1.
        if not use_only_caption_regions or "Visual" not in task.__class__.__name__:
            logger.info("using all regions in the image, "
                        "might cause worse performance due to noise.\n"
                        "Consider setting '--use_only_caption_regions' to True")
            mask_max_value = -2
        else:
            logger.info("'--use_only_caption_regions' set. Reducing number of regions")
            mask_max_value = -1

        self.save_hyperparameters()

        # select the correct model for processing embeddings
        embedding_mapping_class = get_mapping_class(embedding_process_model)
        self.embedding_model = embedding_mapping_class(
            model_hidden_dim=model_hidden_dim
        )
        probe_class = get_probe_class(probe)
        self.probe = probe_class(model_hidden_dim=model_hidden_dim, **probe_group_args)
        self.reporter = task.get_reporter(results_root_dir, mask_max_value)
        self.loss = task.loss

        self.best_val_loss = 99999999
        self.is_best = False

    def forward(self, input_):
        word_representations = self.embedding_model(input_)
        predictions = self.probe(word_representations)
        return predictions

    def training_step(self, batch, batch_idx):
        input_batch, label_batch, length_batch, _ = batch
        predictions = self(input_batch)
        loss, count = self.loss(predictions, label_batch, length_batch)
        self.log("loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        input_batch, label_batch, length_batch, observation_batch = batch
        max_l: int = length_batch.max().detach().cpu().item()  # type: ignore
        input_batch = input_batch[:, :max_l]
        if len(label_batch.shape) == 3:
            label_batch = label_batch[:, :max_l, :max_l]
        else:
            label_batch = label_batch[:, :max_l]
        predictions = self(input_batch)
        loss, count = self.loss(predictions, label_batch, length_batch)
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return loss, predictions, label_batch, length_batch, observation_batch

    def test_step(self, batch, batch_idx):
        input_batch, label_batch, length_batch, observation_batch = batch
        predictions = self(input_batch)
        loss, count = self.loss(predictions, label_batch, length_batch)
        self.log(
            "test_loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        return loss, predictions, label_batch, length_batch, observation_batch

    def _epoch_end(
        self, outputs: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]], split
    ) -> None:
        predictions = [out[1] for out in outputs]
        labels = [out[2] for out in outputs]
        lengths = [out[3] for out in outputs]
        observations = [out[4] for out in outputs]
        report = self.reporter.__call__(
            predictions, labels, lengths, observations, split
        )

        for key, val in report.items():
            if split == "val" and key in ["root_acc", "uuas", "spearmanr_mean_5-50"]:
                add_to_bar = True
            else:
                add_to_bar = False
            self.log(f"{split}_{key}", val, prog_bar=add_to_bar)
        if split == "val":
            assert self.trainer is not None
            if self.trainer.logged_metrics[f"{split}_loss"] < self.best_val_loss:
                self.is_best = True
                self.best_val_loss = self.trainer.logged_metrics[f"{split}_loss"]
                self.log(f"{split}_loss_best_epoch", self.best_val_loss, prog_bar=False)
                for key, val in report.items():
                    self.log(f"{split}_{key}_best_epoch", val, prog_bar=False)
        else:
            self.log(f"{split}_best_epoch", self.current_epoch)

    def validation_epoch_end(
        self, outputs: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
    ) -> None:
        self._epoch_end(outputs, "val")

    def test_epoch_end(
        self, outputs: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
    ) -> None:
        self._epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.probe.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ProbingModel")
        parser.add_argument(
            "--embedding_process_model",
            type=str,
            default="DiskMapping",
            choices=EMBEDDING_MAPPINGS,
            help="What model to use to process the embeddings before probing.",
        )
        parser.add_argument(
            "--use_only_caption_regions",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="By default we use all regions. If you assign this as True, "
            "exclude all region from other captions.",
        )
        return parent_parser
