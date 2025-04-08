CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-2020/judged \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --checkpoint_path colbert-ir/colbertv2.0 \
    --searcher colbert

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-2019/judged \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --checkpoint_path colbert-ir/colbertv2.0 \
    --searcher colbert

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-hard \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --checkpoint_path colbert-ir/colbertv2.0 \
    --searcher colbert

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/dev/small \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --checkpoint_path colbert-ir/colbertv2.0 \
    --searcher colbert

# nohup ./create_colbert_baseline_runs.sh 2>  create_colbert_baseline_runs.out &