import os
import random
import time
import pandas as pd
import numpy as np
import gc
import sys
from itertools import chain
from typing import List, Dict, Any, Optional, Tuple
import torch
from gp.data import load_dms_data, load_experimental_embeddings, load_experimental_data, create_iteration_dataframes, load_config, DictToObject, get_aux_data, get_gp_inputs
from evolvepro.src.utils import pca_embeddings
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from gp.model_gridsearch import top_layer_baseline
from utils import calc_ndcg, calc_toprecall, split_data, group_scores, summarize_scores
from Bio import SeqIO
from transformers import EsmTokenizer  
from scipy.stats import norm
from pathlib import Path

model_class_dict={
    'svm': SVR,
    'rf': RandomForestRegressor,
    'gbt': GradientBoostingRegressor,
    'linear': linear_model.LinearRegression,
}

def param_comb(param_grid):
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_param_combinations = list(product(*param_values))
    

    return param_names, all_param_combinations

# Function to run the directed evolution simulation
def grid_search_proteingym(
    dataset_names: str,
    work_dir_base: str,
    num_simulations: int, 
    top_model: str,
    param_grid: Dict[str, Any],
    shift: int = 100,
    embeddings_file_type: str = 'csv',
    measured_var: str = 'DMS_score',
    embeddings_type_pt: Optional[str] = None,
    model_name: str = 'esm2_t33_650M_UR50D',
    ) -> pd.DataFrame:

    """
    Run the directed evolution simulation.

    Args:
    labels (pd.DataFrame): DataFrame of labels.
    embeddings (pd.DataFrame): DataFrame of embeddings.
    num_simulations (int): Number of simulations to run.
    num_iterations (int): Number of iterations to run.
    num_mutants_per_round (int): Number of mutants to select per round.
    measured_var (str): Measured variable.
    regression_type (str): Type of regression model.
    learning_strategy (str): Learning strategy.
    top_n (int): Number of top variants to consider.
    final_round (int): Number of final round mutants.
    first_round_strategy (str): First round strategy.
    embedding_type (str): Type of embeddings.
    explicit_variants (list): List of explicit variants.

    Returns:
    pd.DataFrame: DataFrame of simulation results.
    """
    # Prepare the data
    #load top_layer config
    cfg = load_config('gp/config.yaml')
    cfg = DictToObject(cfg)
    path = f'data/proteingym/merged.pkl'
    datasets = torch.load(path, weights_only=False)
    proteins = []
    for dataset_name in dataset_names:
        if dataset_name in datasets.keys():
            proteins+=(datasets[dataset_name])
        elif dataset_name =='all':
            proteins = list(chain(*datasets.values()))
        else:
            print(f'Dataset {dataset_name} not found.')
            sys.exit()

    # model param init
    param_names, param_combinations = param_comb(param_grid)
    for combo in param_combinations:
        param_dict = dict(zip(param_names, combo))
        model_id = f"{top_model}-{'-'.join([f'{k}:{v}' for k, v in param_dict.items()])}"
        model_class = model_class_dict[top_model]
        # run 5 rep
        for i in range(1, num_simulations + 1):
            print(f'**********************************************************')
            print(f'=======================Simulation {i}=======================')
            print(f'**********************************************************')
            
            file_paths = [
                f'./output_gridsearch/{top_model}/{model_id}/spearman/rep{i}.csv',
                f'./output_gridsearch/{top_model}/{model_id}/ndcg/rep{i}.csv',
                f'./output_gridsearch/{top_model}/{model_id}/topk_pr/rep{i}.csv'
            ]
            # Check if the file exists, if not, create the directory and the file
            count=0
            for path in file_paths:
                if not os.path.exists(path):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                else:
                    count+=1
            if count==3:
                print(f'Experiment (model: {model_id}, rep: {i}) has been done, skip...')
                continue
            # initialize report saving dict
            reports = {}
            # different datasets at same rep
            for protein in proteins:
                print(f'**********************Current dataset: {protein["name"]}**********************')
                # different training size
                if len(protein['df']) < 100 * num_simulations:
                    sample_idx = int(10 * (i-1))
                else:
                    sample_idx = int(shift * (i-1))
                #print(protein)
                # getting data path
                data_path = f'{work_dir_base}/{protein["name"]}'
                embeddings, labels_true = load_dms_data(protein, model_name, data_path, embeddings_file_type, embeddings_type_pt)
                #print(f'emb:\n{embeddings}\nlabels:\n{labels_true}')

                # Perform mutant selection for the subsequent rounds
                print(f'======================Training Size {100}======================')
                sample_ids = labels_true.iloc[sample_idx:sample_idx+100, :]['variant'].tolist()
                sample_marks= pd.DataFrame({'variant': sample_ids, 'iteration': 0})
                labels_new = pd.merge(labels_true, sample_marks, on='variant', how='left')
                print("Sample including:\n", labels_new[labels_new['iteration']==0])

                # init model
                model = model_class(**param_dict)               
                y_pred_test = top_layer_baseline(
                    iter_train=[0], iter_test=None,
                    embeddings_pd=embeddings, labels_pd=labels_new,
                    measured_var=measured_var, model=model)
                
                #calculate metrics
                train, test = split_data(protein, train_ids=sample_ids)
                #print(train)
                predicts = pd.Series(y_pred_test, index=test['df'].index, name='prediction')
                report, _ = group_scores(train['df'], predicts, test['df'])
                #print(report)
                reports[protein['name']] = report

            
            #saving results
            reports = summarize_scores(reports)
            spearmanr_results = reports['spearmanr']
            ndcg_results = reports['ndcg']
            topk_pr_results = reports['topk_pr']
            
            spearmanr_results.to_csv(f'./output_gridsearch/{top_model}/{model_id}/spearman/rep{i}.csv')
            ndcg_results.to_csv(f'./output_gridsearch/{top_model}/{model_id}/ndcg/rep{i}.csv')
            topk_pr_results.to_csv(f'./output_gridsearch/{top_model}/{model_id}/topk_pr/rep{i}.csv')
            
    return None


if __name__ == "__main__":
    grid_search_dict = {
        'svm':{
            'C':[1, 10, 100],
            'gamma':[1, 0.1, 0.01],
        },
        'rf':{
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 1, 10],
            'min_samples_leaf': [1, 2, 5],
            
        },
        'gbt':{
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 1, 10],
            'min_samples_leaf': [1, 2, 5]
        },
        'linear':{
            'fit_intercept': [True, False],
            'positive': [True, False],
        },
    }
    
    proteingym_subset = ['all']
    models = ['svm', 'rf', 'gbt', 'linear']
    for model in models:
        output_table = grid_search_proteingym(
            dataset_names=proteingym_subset,
            work_dir_base='output_proteingym/dms',
            embeddings_file_type='csv',
            num_simulations=5,
            model_name='esm2_t33_650M_UR50D',
            top_model=model,
            param_grid=grid_search_dict[model],
            )
    
