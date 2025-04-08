import ir_measures
from ir_measures import P, nDCG, R, MRR, MAP
import ir_datasets

DL_EVAL_METRICS_SHORT =[MAP(rel=2), nDCG@10, MRR(rel=2)@10]
BEIR_EVAL_METRICS = [nDCG@10]

BASELINE_RUNS_DICT = {
    'dl19': "msmarco-passage-trec-dl-2019-judged.run",
    'dl20': "msmarco-passage-trec-dl-2020-judged.run",
    'dlhard': "msmarco-passage-trec-dl-hard.run",
    'devsmall': "msmarco-passage-dev-small.run",
}

DATASET_IDS_DICT = {
    'dl19': "msmarco-passage/trec-dl-2019/judged",
    'dl20': "msmarco-passage/trec-dl-2020/judged",
    'dlhard': "msmarco-passage/trec-dl-hard",
    'devsmall': "msmarco-passage/dev/small",
}

IN_DOMAIN_DATASETS = ['dl19', 'dl20', 'dlhard', 'devsmall']
OUT_DOMAIN_DATASETS = ["climate-fever", "dbpedia", "nq", "quora", "scidocs", "trec-covid", "arguana"]


def compute_metrics_from_ir_dataset(dataset_id: str, path_to_run: str, metrics: list, aggregate: bool = True):
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
    dataset = ir_datasets.load(dataset_id) # load dataset
    qrels = dataset.qrels_iter() # get qrels
    
    run = ir_measures.read_trec_run(path_to_run)
    
    if aggregate:
        return ir_measures.calc_aggregate(metrics, qrels, run)
    
    return ir_measures.iter_calc(metrics, qrels, run)
   
    
def get_metrics_for_dataset(mode: str) -> list:
    """
        Get the IR evaluation metrics for a given dataset 

        Args:
            mode (str): dataset string identifier

        Returns:
            list: list of IR evaluation metrics for the dataset
    """
    if mode in IN_DOMAIN_DATASETS:
        metrics = DL_EVAL_METRICS_SHORT
    else:
        metrics = BEIR_EVAL_METRICS
    return metrics
