# Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers
This repository contains the source code used for the experiments presented in the paper "Exploring the Effectiveness of Multi-stage Fine-tuning for Cross-encoder Re-rankers" by Francesca Pezzuti, Sean MacAvaney and Nicola Tonellotto, accepted for publication at ECIR, 2025 - [PDF](https://arxiv.org/abs/2503.22672).

## Citing and Authors
Please, cite our paper if you use this code, a modified version of it, or if you find this repository helpful.

## Usage

### Requirements
You can install the requirements running the following command: 
```
pip install -r requirements.txt
```

### Training datasets

| Dataset | Note | Link |
| --- | --- | --- |
| RankGPT |  top 20 passages retrieved by RankGPT-3.5 for 100k MS MARCO train queries | [RankGPT's official Github repository](https://github.com/sunnweiwei/RankGPT) |
| ColBERTv2 | top 500 passages retrieved by ColBERTv2 for 503k MS MARCO train queries | [Zenodo ColBERTv2 dataset](https://zenodo.org/records/10952882) |


#### Pre-processing of the datasets
To pre-process the training datasets, use the script `preprocessing.py`, specifying as argument the name of the training dataset (we support *colbert* or *rankgpt*).

Please note that:
- The flag to specify the datast is `--dataset`.
- The flag to specify the percentage of data to use as validation is `--evalperc`

### Fine-tuning
Cross encoders can be fine-tuning using a single fine-tuning stage, or two fine-tuning stages in sequence.

#### First stage
To fine-tune a RoBERTa *cross-encoder* with the **knowledge distillation** objective, a *learning rate* of $1e-5$, on *gpu* 2, you can simply run the following command:
```bash
./ft-monoencoder.sh -f 1 -c roberta -g 2 -m KD -l 1e-5
```

Please note that:
- In this command, to use a **contrastive learning** objective, one can use `-c electra`.
- You may notice that flag `-f` set to 1. This indicates that the model is fine-tuned for the first time.
- Using flag `-b`, one can additionally specify the *batch size*.

#### Second stage
To further fine-tune with **knowledge distillation**, a RoBERTa *cross-encoder* previously fine-tuned with **contrastive learning**, using a *learning rate* of $1e-5$, on *gpu* 2, you can simply run the following command:
```bash
./ft-monoencoder.sh -f 2 -c roberta -g 2 -m KD -l 1e-5
```

Please note that:
- In this command, flag `-f 2` is used to tell the main program to automatically search for a model fine-tuned with a first stage of knowledge distillation, whose checkpoint is stored in the `/models/` directory.

### Evaluation
When the fine-tuning is finished, you can use our inference scripts to infer a **document ranking**, and **re-ranking** for queries from the supported datasets.

#### Ranking
To rank with BM25, you can use the script `create_bm25_baseline_run.sh`.

Else, to rank with ColBERTv2, you can use the script `createcolbert_baseline_runs.sh`

#### Re-ranking
As an example, to infer a re-ranking with the *electra* cross-encoder fine-tuned with a single stage, on gpu 2, you can run the following command:
```bash
./inference-monoencoder.sh -f 1 -c electra -m CL -d indomain -g 2
```

## Configuration Files
All the configuration files are stored under directory `/configs/` and are in *yaml* format.