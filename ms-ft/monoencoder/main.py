from typing import Any, Optional, Union

import torch
from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI
from torch.optim import Optimizer

from datamodule import MonoEncoderDataModule

from mono_encoder_module import MonoEncoderModule
from utils.warmup_schedulers import LR_SCHEDULERS, ConstantSchedulerWithWarmup, LinearSchedulerWithWarmup


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class MonoEncoderLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[Union[ConstantSchedulerWithWarmup, LinearSchedulerWithWarmup]] = None,
    ) -> Any:
        if lr_scheduler is None:
            return optimizer

        return [optimizer], [{"scheduler": lr_scheduler, "interval": lr_scheduler.interval}]

    def add_arguments_to_parser(self, parser):
        parser.add_lr_scheduler_args(tuple(LR_SCHEDULERS))
        parser.link_arguments("model.model_name_or_path", "data.model_name_or_path")
        parser.link_arguments("trainer.max_steps", "lr_scheduler.init_args.num_training_steps")


def main():   
    MonoEncoderLightningCLI(
        model_class=MonoEncoderModule,
        datamodule_class=MonoEncoderDataModule,
        save_config_kwargs={"config_filename": "pl_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
