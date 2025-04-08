import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import json
import zipfile
# coding=utf-8
import argparse

SEED = 42

def load_colbert_dataset(fpath: str) -> pd.DataFrame:
    """
        Load the Colbert dataset from the given file path.

        Args:
            fpath (str): Path to the Colbertv2 dataset

        Returns:
            pd.DataFrame: DataFrame containing the Colbert dataset
    """
    df = pd.read_csv(fpath, sep='\t', header=None)
    return df

def _load_rankgpt_outputs(fpath: str) -> object:
    """
        Given a file path, load the rankgpt outputs.

        Args:
            fpath (str): file path to the rankgpt outputs

        Returns:
            object: json object containing the rankgpt outputs
    """
    with open(fpath, 'r') as file:
        data = json.load(file)
    # an example of output is:
        # '[5] > [18] > [14] > [1] > [15] > [3] > [4] > [13] > .... > [19] > [20] > [8] > [17]'
    return data

def load_rankgpt_dataset(zip_fpath: str, jsonl_fname: str, original_dataset_dir: str, dataset_fname: str, proc_dataset_fpath: str) -> pd.DataFrame:
    """
        Preprocess RankGPT data by loading retrieved documents and outputs.

        Args:
            obj (object): _description_

        Returns:
            pd.DataFrame: _description_
    """
    # load dataset and preprocess it
    retrieved_docs_per_query = _load_rankgpt_retrieved_docs_per_query(zip_fpath=zip_fpath, jsonl_fname=jsonl_fname)

    # load outputs from rankgpt
    rankgpt_outputs_fpath = original_dataset_dir+dataset_fname
    rankgpt_outputs = _load_rankgpt_outputs(rankgpt_outputs_fpath)
    
    all_run_entries = [] # list of entries for the dataset
    
    for i, rankgpt_perquery_output in enumerate(rankgpt_outputs): # iterate over the rankgpt ranking lists
        q_num = i + 1
        q_ranked_list = _from_rankgpt_output_to_list(rankgpt_perquery_output, q_num) # convert to a ordered list of docids
        
        for rank, retrieved_docid in enumerate(q_ranked_list): # iterate over the ranked list
            rank = rank + 1
            score = 1/rank # ignored
            true_docid = retrieved_docs_per_query[q_num]["retrieved_docs"][retrieved_docid]
            true_qid = retrieved_docs_per_query[q_num]["qid"]
            csv_entry =  [true_qid, 0, true_docid, rank, 1/rank, "RankGPT3.5"] # trec format
            all_run_entries.append(csv_entry)
            
        if q_num % 10000 == 0:
            print(f"Processed {q_num} queries")

    dataset_df = pd.DataFrame(all_run_entries) # create dataframe for the runs

    # remove queries with less than 20 scored documents
    #grouped_counts = dataset_df.groupby([0]).size().reset_index(name='Count')
    #filtered_groups = grouped_counts[grouped_counts['Count'] < 20] # search queries with less than 20 scored documents

    save_dataset_to_file(dataset_df, proc_dataset_fpath) # save dataset to file

    return dataset_df

def _load_rankgpt_retrieved_docs_per_query(zip_fpath: str, jsonl_fname: str) -> dict:
    """
        Load the rankgpt dataset from the zip file and the jsonl file

        Args:
            zip_fpath (str): path to the zip file containing the dataset
            jsonl_fname (str): name of the jsonl file

        Returns:
            dict: dictionary containing the rankgpt dataset. Keys are the query numbers,
                  values are the dictionaries containing pairs (rank: docid)
    """
    retrieved_docs_per_query = {}
    with zipfile.ZipFile(zip_fpath, 'r') as z: # open zip
        with z.open(jsonl_fname) as jsonl_file: # open jsonl
            q_num = 0
            for line in jsonl_file: # read jsonl line by line
                q_num = q_num + 1
                json_obj = json.loads(line.decode('utf-8')) # load json object associated to the query
                for doc in json_obj["retrieved_passages"]: # iterate over the retrieved passages
                    if q_num not in retrieved_docs_per_query:
                        retrieved_docs_per_query[q_num] = {"qid": json_obj["query_id"], "retrieved_docs": {}}
                    
                    rank = doc["rank"]
                    docid = doc["docid"]
                    retrieved_docs_per_query[q_num]["retrieved_docs"][rank] = docid
    return retrieved_docs_per_query


def _from_rankgpt_output_to_list(formatted_string: str, q_num: int) -> list:
    """ 
        Converts the formatted string from rankgpt output (a run) into an ordered list of document identifiers.

        Args:
            formatted_string (str): output from rankgpt
            q_num (int): query number (used for printing purposes)

        Returns:
            list: oredered list of document identifiers (ranked documents)
    """
    list_of_strings = formatted_string.replace('[', '').replace(']', '').split(' > ')
    list_of_integers = []
    for num in list_of_strings: # form list of strings, to list of int
        try:
            list_of_integers.append(int(num))
        except ValueError:
            print(f"Invalid entry for {q_num}-th query: ", formatted_string)
            return [] # skip the invalid entry
        
    if len(list_of_integers) != 20:
        print(f"Invalid entry for {q_num}-th query: ", formatted_string)
        return [] # skip the invalid entry
    return list_of_integers


def split_train_eval(to_split_df: pd.DataFrame, ranklist_len: int, eval_perc: float = 1.0, seed: int = SEED) -> tuple:
    """
        Splits the dataset into training and validation sets

    Args:
        to_split_df (pd.DataFrame): dataset to be split
        ranklist_len (int): number of ranked documents per query
        eval_perc (float, optional): Percentage of data to use as validation set. Defaults to 1.0.
        seed (int, optional): seed for the random split.

    Returns:
        tuple: training and validation datasets
    """
    qids = to_split_df.iloc[:, 0].unique()
    print(f"Number of queries in the dataset: {len(qids)}")
   
    train_df, eval_df = train_test_split(qids, test_size=eval_perc, random_state=seed)

    print(f"Training set shape: {train_df.shape[0]}")
    print(f"Evaluation set shape: {eval_df.shape[0]}")
    
    train_dataset = to_split_df[to_split_df.iloc[:, 0].isin(train_df)]
    print("Queries selected in the training dataset: ", int(train_dataset.shape[0]/ranklist_len))
    
    eval_dataset = to_split_df[to_split_df.iloc[:, 0].isin(eval_df)]
    print("Queries selected in the evaluation dataset: ", int(eval_dataset.shape[0]/ranklist_len))
    
    return train_dataset, eval_dataset

def save_dataset_to_file(dataset_df: pd.DataFrame, fpath: str) -> None:
    """
        Save pre-processed dataset to file

        Args:
            dataset_df (pd.DataFrame): dataframe storing the dataset to be saved
            fpath (str): output file path
    """
    dataset_df.to_csv(fpath, index=False, sep='\t', header=False)

def main():
    SUPPORTED_DATASETS = ["colbert", "rankgpt"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=SUPPORTED_DATASETS, help="Collection to pre-process.")
    parser.add_argument("--evalperc", type=float, default=0.01, choices=SUPPORTED_DATASETS, help="Percentage of data to be used for validation.")
    args = parser.parse_args() # parse arguments

    dataset_str_id = args.dataset
    
    DATA_DIR = "./../../data/"
    TRAINING_RUNS_DIR = DATA_DIR + "runs/training/"

    if dataset_str_id == "colbert":
        dataset_fname = "__colbert__msmarco-passage-train-judged.run"
        dataset_dir = TRAINING_RUNS_DIR + "colbert/"
        train_dataset_dir =  TRAINING_RUNS_DIR + "colbert-train/"
        eval_dataset_dir = TRAINING_RUNS_DIR + "/colbert-eval/"
        dataset_fpath = dataset_dir + dataset_fname
        preproc_fname = dataset_fname
        ranklist_len = 500

        # load dataset
        dataset_df = load_colbert_dataset(dataset_fpath)
        
    elif dataset_str_id == "rankgpt":
        dataset_fname = 'marco-train-100k-gpt3.5.json'
        dataset_dir = TRAINING_RUNS_DIR + "rankgpt-sun/"
        eval_dataset_dir = TRAINING_RUNS_DIR + "rankgpt-sun-eval/"
        train_dataset_dir = TRAINING_RUNS_DIR + "rankgpt-sun-train/"
        preproc_fname = "__rankgptsun__msmarco-passage-train-judged.run"
        ranklist_len = 20
    
        zip_fpath = dataset_dir+ "marco-train-100k.jsonl.zip"
        jsonl_fname = "marco-train-100k.jsonl"

        # load dataset
        dataset_df = load_rankgpt_dataset(zip_fpath=zip_fpath, jsonl_fname=jsonl_fname, original_dataset_dir=dataset_dir,
                                          dataset_fname=dataset_fname, proc_dataset_fpath=dataset_dir+preproc_fname)

    train_dataset, eval_dataset = split_train_eval(to_split_df=dataset_df, ranklist_len=ranklist_len, eval_perc=args.evalperc)

    eval_fpath = eval_dataset_dir + dataset_fname
    train_fpath = train_dataset_dir + dataset_fname

    save_dataset_to_file(eval_dataset, eval_fpath)
    save_dataset_to_file(train_dataset, train_fpath)