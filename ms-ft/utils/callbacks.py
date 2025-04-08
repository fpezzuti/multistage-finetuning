import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import ir_datasets
import pandas as pd
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback

from utils.ir_dataset_utils import load as load_ir_dataset
from utils.ir_dataset_utils import DASHED_DATASET_MAP

RUN_HEADER = ["query", "q0", "docid", "rank", "score", "system"] # run file header (trec format)

class PredictionWriter(BasePredictionWriter):
    def __init__(self, overwrite: bool = False) -> None:
        super().__init__("batch")
        self.overwrite = overwrite

    def get_run_path(self, trainer: Trainer, pl_module: LightningModule, dataset_idx: int) -> Path:
        assert hasattr(trainer, "datamodule")
        assert hasattr(trainer, "ckpt_path")
        ckpt_path = Path(trainer.ckpt_path)
        datamodule = trainer.datamodule
        ir_dataset_path = datamodule.predict_ir_dataset_paths[dataset_idx]
        ir_dataset = load_ir_dataset(ir_dataset_path)
        original_ir_dataset_id = re.sub(r"__.+__", "", ir_dataset.dataset_id().split("/")[-1])
        original_ir_dataset = ir_datasets.load(DASHED_DATASET_MAP[original_ir_dataset_id])
        dataset_id = original_ir_dataset.dataset_id().replace("/", "-")
        filename = (f"{Path(ir_dataset_path).parent.name}_{datamodule.depth}_{dataset_id}.run")
        run_file_path = ckpt_path.parent.parent / "runs" / filename
        return run_file_path

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # skips writing to run files if they exist
        assert hasattr(trainer, "datamodule")
        datamodule = trainer.datamodule
        for dataset_idx, dataset in enumerate(datamodule.predict_datasets):
            run_file_path = self.get_run_path(trainer, pl_module, dataset_idx)
            if not self.overwrite and run_file_path.exists():
                if datamodule.num_workers != 0:
                    warnings.warn("skipping already predicted runs only works if num_workers=0")
                else:
                    dataset.run_query_ids = dataset.run_query_ids.drop(
                        dataset.run_qids.index
                    )  # TODO fix with new base data parsing

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        run_file_path = self.get_run_path(trainer, pl_module, dataloader_idx)
        doc_ids = batch["doc_ids"]
        query_ids = batch["query_id"]
        scores = [float(logit.item()) for logits in prediction for logit in logits]
        query_ids = [
            query_id
            for batch_idx, query_id in enumerate(query_ids)
            for _ in range(len(doc_ids[batch_idx]))
        ]
        doc_ids = [doc_id for doc_ids in doc_ids for doc_id in doc_ids]
        run_df = pd.DataFrame(
            zip(query_ids, doc_ids, scores), columns=["query", "docid", "score"]
        )
        run_df = run_df.sort_values(["query", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("query")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        run_df["q0"] = 0
        run_df["system"] = "bilist_encoder"
        run_df = run_df[RUN_HEADER]
        run_file_path.parent.mkdir(exist_ok=True)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"
        run_df.to_csv(run_file_path, header=False, index=False, sep="\t", mode=mode)

class EarlyStoppingValidationStep(Callback):
    def __init__(self, monitor="eval/val_loss", patience=10, verbose=False, mode='min'):
        super().__init__()
        self.monitor =monitor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.stopped_epoch = 0
        self.early_stop = False

        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode '{self.mode}' is not in ['min', 'max']")
        
        self.is_min_mode = self.mode == 'min'
        
    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics[self.monitor]
    
        if self.is_min_mode:
            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                self.wait = 0 # reset wait
            else:
                self.wait +=  1 # increase wait counter
        else:
            # if the current score is better than the best score
            if self.best_score is None or current_score > self.best_score:
                self.best_score = current_score
                self.wait = 0 # reset wait
            else:
                self.wait += 1 # increase wait counter
                
        # check if we should stop early
        if self.wait >= self.patience:
            self.early_stop = True
            trainer.should_stop = True
            if self.verbose:
                print(f"Early stopping triggered at step {trainer.global_step}")