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
from gp.model import top_layer as gp_optim
from gp.model import top_layer_baseline
from utils import calc_ndcg, calc_toprecall, split_data, group_scores, summarize_scores
from Bio import SeqIO
from transformers import EsmTokenizer  
from scipy.stats import norm
from pathlib import Path

# Function to run the directed evolution simulation
def directed_evolution_simulation(
    dataset_name: str,
    work_dir_base: str,
    num_simulations: int, 
    num_groups: int, 
    shift: int = 100,
    embeddings_file_type: str = 'csv',
    measured_var: str = 'DMS_score',
    embeddings_type_pt: Optional[str] = None,
    num_mutants_per_group: int = 20, 
    top_model: str = 'gp_optim',
    model_name: str = 'esm2_t33_650M_UR50D',
    device: str = 'cuda:0',
    ) -> pd.DataFrame:

    # Prepare the data
    #load top_layer config
    cfg = load_config('gp/config.yaml')
    cfg = DictToObject(cfg)
    path = f'data/proteingym/merged.pkl'
    datasets = torch.load(path, weights_only=False)
    if dataset_name in datasets.keys():
        proteins = datasets[dataset_name]
    else:
        proteins = chain(*datasets.values())
        if dataset_name =='all':
            proteins = list(proteins)
    
    for i in range(1, num_simulations + 1):
        print(f'**********************************************************')
        print(f'=======================Simulation {i}=======================')
        print(f'**********************************************************')
        
        j = 1    
        while j <= num_groups:
            file_paths = [
                f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/spearman/rep{i}.csv',
                f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/ndcg/rep{i}.csv',
                f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/topk_pr/rep{i}.csv'
            ]
            # Check if the file exists, if not, create the directory and the file
            count=0
            for path in file_paths:
                if not os.path.exists(path):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                else:
                    count+=1
            if count==3:
                print(f'Experiment (model: {top_model}, rep: {i}, training_size: {num_mutants_per_group*j}) has been done, skip...')
                j+=1
                continue
            # initialize report saving dict
            reports = {}
            # different datasets at same training size
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

                # Load aux data for GP
                aux = get_aux_data(
                    labels_pd=labels_true, 
                    load_path=data_path,
                    dataset_name=protein["name"],
                )
                #print(f'emb:\n{embeddings}\naux:\n{aux}\nlabels:\n{labels_true}')

                # prepare GP inputs
                gp_inputs = get_gp_inputs(
                    dataset_name=protein["name"],
                    load_path=data_path,
                    cfg=cfg,
                    seq=protein['wild_type'],
                    use_mpnn=True,
                )
                
                # Initialize the variables
                single_local_spearman = None
                single_cross_spearman = None
                single_rest_spearman = None
                multi_combined_spearman = None
                multi_cross_spearman = None
                multi_rest_spearman = None
                all_rest_spearman = None
                single_local_ndcg = None
                single_cross_ndcg = None
                single_rest_ndcg = None
                multi_combined_ndcg = None
                multi_cross_ndcg = None
                multi_rest_ndcg = None
                all_rest_ndcg = None
                single_local_topk_pr = None
                single_cross_topk_pr = None
                single_rest_topk_pr = None
                multi_combined_topk_pr = None
                multi_cross_topk_pr = None
                multi_rest_topk_pr = None
                all_rest_topk_pr = None

            
                # Perform mutant selection for the subsequent rounds
                print(f'======================Training Size {j*num_mutants_per_group}======================')
                sample_ids = labels_true.iloc[sample_idx:sample_idx+j*num_mutants_per_group, :]['variant'].tolist()
                sample_marks= pd.DataFrame({'variant': sample_ids, 'iteration': 0})
                labels_new = pd.merge(labels_true, sample_marks, on='variant', how='left')
                print("Sample including:\n", labels_new[labels_new['iteration']==0])


                if top_model == 'gp_optim':
                    y_pred_test = gp_optim(
                        iter_train=[0], iter_test=None,
                        embeddings_pd=embeddings, labels_pd=labels_new, aux_pd=aux, gp_inputs=gp_inputs, cfg=cfg,
                        measured_var=measured_var, final_round=num_mutants_per_group, total_iteration=num_groups, current_iteration=j, device=device, use_TS=False, is_ALDE=False)
                else:
                    y_pred_test = top_layer_baseline(
                        iter_train=[0], iter_test=None,
                        embeddings_pd=embeddings, labels_pd=labels_new,
                        measured_var=measured_var, regression_type=top_model, final_round=num_mutants_per_group)
                
                #calculate metrics
                train, test = split_data(protein, train_ids=sample_ids)
                #print(train)
                predicts = pd.Series(y_pred_test, index=test['df'].index, name='prediction')
                report, _ = group_scores(train['df'], predicts, test['df'])
                print(report)
                reports[protein['name']] = report
            #print(reports)
            
            #saving results
            reports = summarize_scores(reports)
            spearmanr_results = reports['spearmanr']
            ndcg_results = reports['ndcg']
            topk_pr_results = reports['topk_pr']
            
            spearmanr_results.to_csv(f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/spearman/rep{i}.csv')
            ndcg_results.to_csv(f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/ndcg/rep{i}.csv')
            topk_pr_results.to_csv(f'./output_proteingym/dms_results_proteingym/{top_model}/ts_{num_mutants_per_group*j}/topk_pr/rep{i}.csv')
            j += 1
            
    return None


if __name__ == "__main__":
    output_table = directed_evolution_simulation(
        dataset_name='all',
        work_dir_base='output_proteingym/dms',
        embeddings_file_type='csv',
        num_simulations=5,
        num_groups=5,
        num_mutants_per_group=20,
        model_name='esm2_t33_650M_UR50D',
        device='cuda:0',
        top_model='gp_optim', # 'svm', 'randomforest', 'gradientboosting', 'linear'
        )
    
