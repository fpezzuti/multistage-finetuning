from typing import Any, Dict

import pandas as pd
from trectools import TrecEval, TrecQrel, TrecRun


ADHOC_METRICS = {"NDCG", "MRR", "UNJ", "rNDCG", "P"}

def evaluate_run(run_df: pd.DataFrame, qrels_df: pd.DataFrame, metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """ Evaluate a run file using the given metrics.

    Args:
        run_df (pd.DataFrame): run file dataframe
        qrels_df (pd.DataFrame): qrels dataframe
        metrics (Dict[str, Dict[str, Any]]): metrics to evaluate the run file

    Returns:
        pd.DataFrame: dataframe of the evaluated metrics
    """
    run_df = run_df.rename({"query_id": "query", "Q0": "q0", "doc_id": "docid", "run_name": "system"}, axis=1,)
    adhoc_metrics = {
        metric: kwargs
        for metric, kwargs in metrics.items()
        if metric.split("@")[0] in ADHOC_METRICS
    }

    adhoc_values = evaluate_adhoc(adhoc_metrics, run_df, qrels_df)
    values = pd.concat([adhoc_values], axis=1).fillna(0)
    return values


def evaluate_adhoc(full_metrics: Dict[str, Any], run_df: pd.DataFrame, qrels_df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        full_metrics (Dict[str, Any]): dict of metrics to evaluate
        run_df (pd.DataFrame): dataframe of the run file
        qrels_df (pd.DataFrame): dataframe of the qrels file

    Returns:
        pd.DataFrame: dataframe of the evaluated metrics
    """
    run = TrecRun()
    qrels = TrecQrel()
    run.run_data = run_df
    qrels_df = qrels_df.groupby(["query", "docid"])["rel"].max().reset_index()
    qrels.qrels_data = qrels_df
    trec_eval = TrecEval(run, qrels)
    metric_to_func = {
        "NDCG": "get_ndcg",
        "rNDCG": "get_ndcg",
        "MRR": "get_reciprocal_rank",
        "UNJ": "get_unjudged",
        "P": "get_precision"
    }
    dfs = []
    for full_metric, kwargs in full_metrics.items():
        metric, depth = full_metric.split("@")
        depth = depth.split("_")[0]
        depth = int(depth)
        func_name = metric_to_func[metric]
        func = getattr(trec_eval, func_name)
        df = func(depth, per_query=True, **kwargs)
        df = df.rename(lambda x: full_metric, axis=1)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df