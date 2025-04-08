import ir_measures
from ir_measures import *

from scipy import stats
import numpy as np
import pandas as pd

from utils.ir_evaluation import compute_metrics_from_ir_dataset

def compute_metrics_for_ttest(dataset_id: str, path_to_run: str, metrics) -> list:
    """
        Compute IR evaluation metrics for a given dataset and run file.
        If aggreagate is True, the function returns the aggregated metrics,
        otherwise it returns the metrics for each query.
        
        Args:
            dataset_id (str): identifier of the dataset
            path_to_run (str): path to the run file
            metrics (list): list of IR evaluation metrics
            aggregate (bool, optional): aggregate per-query metric. Defaults to True.
        
        Returns:
            _type_: aggregated or per-query metrics computed over the run file.
    """
    
    results = compute_metrics_from_ir_dataset(dataset_id, path_to_run, metrics, aggregate=False)

    # prepare the dataframe for the run
    df_results = pd.DataFrame(results, columns=['qid', 'metric', 'metric_value'])

    df_results.replace(metrics, np.arange(len(metrics)), inplace=True) # transform metric name to id (int)
    df_results.sort_values(['metric', 'qid'], inplace=True) # sort by metric id (int)
    df_results.reset_index(drop=True, inplace=True)
 
    return df_results['metric_value'].to_list()

def twosided_paired_ttest(baseline_res: pd.DataFrame, run_res: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """ Perform two-sided paired t-test for each metric given two dataframes containing IR metrics.

    Args:
        baseline_res (pd.DataFrame): dataframe storing the IR metrics for the baseline run
        run_res (pd.DataFrame): dataframe storing the IR metrics for the run
        metrics (list): list of IR evaluation metrics

    Returns:
        pd.DataFrame: dataframe of t-ttest p-values for each metric
    """
    n_queries = int(len(baseline_res) / len(metrics))

    ttest_res_dict = {}
    
    for metric_id, metric in enumerate(metrics): # iterate over metrics
        start_id = metric_id*n_queries # start index for the metric
        end_id = start_id + n_queries # end index for the metric
        base_vals = baseline_res[start_id:end_id]
        run_vals = run_res[start_id:end_id]

        assert len(base_vals) == n_queries and len(run_vals) == n_queries
        
        # perform two-sided paired t-test
        ttest_res = stats.ttest_rel(base_vals, run_vals, nan_policy='raise')
        ttest_res_dict[str(metric)] = ttest_res.pvalue
    
    ttest_df = pd.DataFrame.from_dict(ttest_res_dict.items()).T
    ttest_df.columns = [str(metric) for metric in metrics] # add header
    ttest_df = ttest_df[1:] # drop old header
   
    return ttest_df

def evaluate_significance_difference(run_path: str, baseline_run_path: str, dataset_id: str, metrics: list) -> pd.DataFrame:
    """ Evaluate the significance difference between two runs given a set of metrics.

    Args:
        run_path (str): path to run file
        baseline_run_path (str): path to baseline run file
        dataset_id (str): string identifier of the dataset
        metrics (list): list of IR evaluation metrics to be evaluated and tested

    Returns:
        dataframe: dataframe of t-test p-values for each metric
    """
    base_res = compute_metrics_for_ttest(dataset_id, baseline_run_path, metrics=metrics)
    run_res = compute_metrics_for_ttest(dataset_id, run_path, metrics=metrics)
    return twosided_paired_ttest(base_res, run_res, metrics)