from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import os
import ir_datasets
import lightning.pytorch as pl
import pandas as pd
import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.ir_dataset_utils import DASHED_DATASET_MAP
from utils import loss_utils
from utils.validation_utils import evaluate_run
from mono_encoder import MonoEncoderClassFactory
from lightning.pytorch.utilities import grad_norm

def parse_model_name(model_name_or_path: str) -> str:
    supported_crossencoders = {"roberta": "monoRoberta", "electra": "monoElectra"}    
    supported_single_stages = ["CL", "KD"]
    
    cross_encoder = "monoEncoder"
    
    ft = None
    
    for crossencoder_name in supported_crossencoders:
        if crossencoder_name in model_name_or_path:
                cross_encoder = supported_crossencoders[crossencoder_name]
                break

    for last_stage_name in supported_single_stages:
        if last_stage_name in model_name_or_path:
            if "ft1" in model_name_or_path:
                ft = f"{last_stage_name}"
            elif "ft2" in model_name_or_path:
                for previous_stage_name in supported_single_stages:
                    if previous_stage_name not in model_name_or_path:
                        ft = f"{previous_stage_name}{last_stage_name}"
            else:
                raise ValueError("Model name not supported.")
            return cross_encoder + "-" + ft

    
class MonoEncoderModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        depth: int = 100,
        freeze_position_embeddings: bool = False,
        loss_function: loss_utils.LossFunc = loss_utils.RankNet(),
        compile_model: bool = True,
        use_flash: bool = True,
        fill_random_docs: bool = True,
    ) -> None:
        super().__init__()
        self.loss_function = loss_function
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        model_class = transformers.AutoModelForSequenceClassification._model_mapping[
            type(config)
        ]
        MonoEncoder = MonoEncoderClassFactory(model_class)
        self.mono_encoder = MonoEncoder.from_pretrained(
            model_name_or_path,
            depth=depth,
            use_flash=use_flash,
            fill_random_docs=fill_random_docs,
        )
        if freeze_position_embeddings:
            for name, param in self.mono_encoder.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False
                    break

        if compile_model:
            torch.compile(self.mono_encoder)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.model_name = parse_model_name(model_name_or_path)
        self.runs_dir = f"./../../data/runs/evaluation/{self.model_name}/"
        if "ft1" in model_name_or_path:
            stage = "ft1"
        elif "ft2" in model_name_or_path:
            stage = "ft2"
        else:
            stage = "noft"
        self.run_prefix = self.mono_encoder.encoder_name + "-" + stage + f"-"
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SequenceClassifierOutput:
        return self.mono_encoder(
            input_ids, attention_mask=attention_mask
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]
        out = self.forward(input_ids, attention_mask)

        logits = torch.nn.utils.rnn.pad_sequence(
            torch.split(out.logits.squeeze(1), num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            torch.split(batch["labels"], num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        
        loss = self.loss_function.compute(logits, labels, None)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> List[torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]
        out = self.forward(input_ids, attention_mask)
        logits = torch.split(out.logits.squeeze(1), num_docs)
        return logits

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        dataset_name = ""
        first_stage = ""
        try:
            ir_dataset_path = Path(
                self.trainer.datamodule.val_ir_dataset_paths[dataloader_idx]
            )
            dataset_name = ir_dataset_path.name[
                : -len("".join(ir_dataset_path.suffixes))
            ]
            first_stage = ir_dataset_path.parent.name
        except RuntimeError:
            return

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]

        out = self.forward(input_ids, attention_mask)
        
        logits = torch.nn.utils.rnn.pad_sequence(
            torch.split(out.logits.squeeze(1), num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            torch.split(batch["labels"], num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        
        
        val_loss = self.loss_function.compute(logits, labels, None)
        self.log("eval/val_loss", val_loss, prog_bar=True)
        
        logits = out.logits.view(-1).tolist()
        
        query_ids = [
            query_id
            for query_idx, query_id in enumerate(batch["query_id"])
            for _ in range(num_docs[query_idx])
        ]
        doc_ids = [doc_id for doc_ids in batch["doc_ids"] for doc_id in doc_ids]

        self.validation_step_outputs.append(
            (
                f"{first_stage}/{dataset_name}",
                {"score": logits, "query_id": query_ids, "doc_id": doc_ids},
            )
        )
        
    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        dataset_name = ""
        first_stage = ""
        try:
            ir_dataset_path = Path(
                self.trainer.datamodule.test_ir_dataset_paths[dataloader_idx]
            )
            dataset_name = ir_dataset_path
        except RuntimeError:
            return

        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]

        out = self.forward(input_ids, attention_mask)
        logits = out.logits.view(-1).tolist()
        query_ids = [
            query_id
            for query_idx, query_id in enumerate(batch["query_id"])
            for _ in range(num_docs[query_idx])
        ]
        doc_ids = [doc_id for doc_ids in batch["doc_ids"] for doc_id in doc_ids]

        self.test_step_outputs.append(
            (
                f"{first_stage}/{dataset_name}",
                {"score": logits, "query_id": query_ids, "doc_id": doc_ids},
            )
        )

    def on_validation_epoch_end(self) -> None:
        aggregated = defaultdict(lambda: defaultdict(list))
        for dataset, value_dict in self.validation_step_outputs:
            for key, value in value_dict.items():
                aggregated[dataset][key].extend(value)

        self.validation_step_outputs.clear()

        for dataset, values in aggregated.items():
            if "trec-dl-2019" in dataset:
                dataset_id = "msmarco-passage-trec-dl-2019-judged"
                dataset_name = "dl19"
            elif "trec-dl-2020" in dataset:
                dataset_id = "msmarco-passage-trec-dl-2020-judged"
                dataset_name = "dl20"
            elif "trec-dl-hard" in dataset:
                dataset_id = "msmarco-passage-trec-dl-hard"
                dataset_name = "dlhard"
            elif "dev-small" in dataset:
                dataset_id = "msmarco-passage-dev-small"
                dataset_name = "devsmall"
            elif "colbert" or "rankgpt" in dataset:
                dataset_id = "msmarco-passage-train-judged"
                dataset_name = "colbert" if "colbert" in dataset else "rankgpt"
            else:
                print(f"Dataset id of dataset={dataset} not available for testing. Please implement it.")
                print(f"Skipped write.")
                continue
                
            if "bm25" in dataset:
                first_stage = "bm25"
            elif "colbert" in dataset:
                first_stage = "colbert"
            elif "tasb" in dataset:
                first_stage = "tasb"
            elif "contriever" in dataset:
                first_stage = "contriever"
            elif "rankgpt" in dataset:
                first_stage = "rankgpt"
            else:
                raise ValueError('Unsupported first stage.')
            
            run_name = self.run_prefix + first_stage + "-" + dataset_name
            
            run = pd.DataFrame(values)
            run["rank"] = run.groupby("query_id")["score"].rank(
                ascending=False, method="first"
            )
            run = run.sort_values(
                ["query_id", "score"], ascending=[True, False]
            ).reset_index(drop=True)
            run["Q0"] = "0"
            run["run_name"] = run_name
            
            #dataset_id = dataset.split("/")[1]
            qrels = pd.DataFrame(
                ir_datasets.load(DASHED_DATASET_MAP[dataset_id]).qrels_iter()
            )
            qrels = qrels.rename(
                {"doc_id": "docid", "relevance": "rel", "query_id": "query"}, axis=1
            )
            missing_qids = set(qrels["query"]) - set(run["query_id"])
            if missing_qids:
                qrels = qrels[~qrels["query"].isin(missing_qids)]
            metrics = {
                "P@8": {},
                "NDCG@1": {},
                "NDCG@5": {},
                "NDCG@10": {},
                "NDCG@100": {},
                "MRR@10": {},
                "MRR@100": {},
            }

            values = evaluate_run(run, qrels, metrics)
            values = values.mean()
            print("****** SCORES: ********\n", values, "**************")
            for metric, value in values.items():
                self.log(f"eval/{metric}", value)
                
            score_var = run['score'].var()
            score_mean = run['score'].mean()
            
            self.log("eval/score_var", score_var, prog_bar=False)
            self.log("eval/score_mean", score_mean, prog_bar=False)
                
    def on_test_epoch_end(self) -> None:
        print("Test inference completed.")
        aggregated = defaultdict(lambda: defaultdict(list))
        for dataset, value_dict in self.test_step_outputs:
            for key, value in value_dict.items():
                aggregated[dataset][key].extend(value)

        self.test_step_outputs.clear()

        for dataset, values in aggregated.items():
            if "trec-dl-2019" in dataset:
                dataset_id = "msmarco-passage-trec-dl-2019-judged"
                dataset_name = "dl19"
            elif "trec-dl-2020" in dataset:
                dataset_id = "msmarco-passage-trec-dl-2020-judged"
                dataset_name = "dl20"
            elif "trec-dl-hard" in dataset:
                dataset_id = "msmarco-passage-trec-dl-hard"
                dataset_name = "dlhard"
            elif "dev-small" in dataset:
                dataset_id = "msmarco-passage-dev-small"
                dataset_name = "devsmall"
            elif "climate-fever" in dataset:
                dataset_id = "beir-climate-fever"
                dataset_name = "climate-fever"
            elif "dbpedia" in dataset:
                dataset_id = "beir-dbpedia-entity-test"
                dataset_name = "dbpedia"
            elif "nq" in dataset:
                dataset_id = "beir-nq"
                dataset_name = "nq"
            elif "quora" in dataset:
                dataset_id = "beir-quora-test"
                dataset_name = "quora"
            elif "scidocs" in dataset:
                dataset_id = "beir-scidocs"
                dataset_name = "scidocs"
            elif "trec-covid" in dataset:
                dataset_id = "beir-trec-covid"
                dataset_name = "trec-covid"
            elif "arguana" in dataset:
                dataset_id = "beir-arguana"
                dataset_name = "arguana"
            else:
                print(f"Dataset id of dataset={dataset} not available for testing. Please implement it.")
                print(f"Skipped write.")
                continue
                
            if "bm25" in dataset:
                first_stage = "bm25"
            elif "colbert" in dataset:
                first_stage = "colbert"
            elif "tasb" in dataset:
                first_stage = "tasb"
            elif "contriever" in dataset:
                first_stage = "contriever"
            else:
                raise ValueError('Unsupported first stage.')
                
            run_name = self.run_prefix + first_stage + "-" + dataset_name
            run = pd.DataFrame(values)
            run["rank"] = run.groupby("query_id")["score"].rank(
                ascending=False, method="first"
            ).map(int)
            run = run.sort_values(
                ["query_id", "score"], ascending=[True, False]
            ).reset_index(drop=True)
            run["Q0"] = "0"
            run["run_name"] = run_name
            
            # reorder cols to met the TREC-format
            run = run[["query_id", "Q0", "doc_id", "rank", "score", "run_name"]]
            
            output_path = self.runs_dir + run_name + ".run"
            print(f"Saving test predictions to file: {output_path}")
            os.makedirs(self.runs_dir, exist_ok=True)
            run.to_csv(output_path, index=False, sep='\t', header=False)
            print(f"Saved test predictions on dataset={dataset} to {output_path}.")
        
            qrels = pd.DataFrame(
                ir_datasets.load(DASHED_DATASET_MAP[dataset_id]).qrels_iter()
            )
            qrels = qrels.rename(
                {"doc_id": "docid", "relevance": "rel", "query_id": "query"}, axis=1
            )
            missing_qids = set(qrels["query"]) - set(run["query_id"])
            if missing_qids:
                qrels = qrels[~qrels["query"].isin(missing_qids)]
            metrics = {
                "P(rel=2)@10": {},
                "NDCG@1": {},
                "NDCG@5": {},
                "NDCG@10": {},
                "NDCG@10_UNJ": {"removeUnjudged": True},
                "UNJ@10": {},
            }

            values = evaluate_run(run, qrels, metrics)
            values = values.mean()
            print(f"****** SCORES: on {dataset_name}********\n{values}\n**************")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            step = self.trainer.global_step
            self.mono_encoder.config.save_step = step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / f"huggingface_checkpoint_step{step}"
            self.mono_encoder.save_pretrained(save_path)
            try:
                self.trainer.datamodule.tokenizer.save_pretrained(save_path)
            except:
                pass
            
    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self, norm_type=2)
        total = norms["grad_2.0_norm_total"]
        self.log_dict({"grad_2.0_norm_total": total})