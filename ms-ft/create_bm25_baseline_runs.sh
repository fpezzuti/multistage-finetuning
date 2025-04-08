CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-2020/judged \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --searcher bm25

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-2019/judged \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --searcher bm25

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/trec-dl-hard \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --searcher bm25

CUDA_VISIBLE_DEVICES=$1 python set_encoder/data/create_baseline_run.py \
    --ir_datasets msmarco-passage/dev/small \
    --run_dir ./../../data/baseline-runs \
    --index_dir ./../../data/indexes \
    --searcher bm25

# nohup ./create_bm25_baseline_runs.sh 2>  create_bm25_baseline_runs.out &