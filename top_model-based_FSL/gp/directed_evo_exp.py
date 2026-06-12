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
def directed_evolution_exp_gp_multi(
    protein_name: str,
    round_name: str,
    work_dir_base: str,
    embeddings_base_path: str,
    embeddings_file_names: List[str],
    round_base_path: str,
    round_file_names: List[str],
    wt_fasta_path: str,
    output_dir = None,
    num_mutants: int = 20, 
    postfix: str = '_XXXX',
    device: str = 'cuda:0',
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
    
    # 1.load emb data
    embeddings_list = []
    for i, file_name in enumerate(embeddings_file_names):
        embedding = load_experimental_embeddings(embeddings_base_path, file_name)
        if i > 0:  # If not the first file
            embedding = embedding[~embedding.index.isin(['WT', 'WT Wild-type sequence'])]
        embeddings_list.append(embedding)
    
    embeddings = pd.concat(embeddings_list)
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # 2.load exp data
    all_experimental_data = []
    for round_file_name in round_file_names:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path, False)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    
    # 3.create iteration dataframes
    iteration, labels = create_iteration_dataframes(all_experimental_data, embeddings.index)

    # 4.load aux data for GP
    data_path = f'{work_dir_base}'
    aux = get_aux_data(
        labels_pd=labels, 
        load_path=data_path,
        dataset_name=protein_name,
    )
    #print(f'emb:\n{embeddings}\naux:\n{aux}\nlabels:\n{labels_true}')

    # 5.prepare GP inputs
    seq = str(SeqIO.read(wt_fasta_path, "fasta").seq)
    gp_inputs = get_gp_inputs(
        dataset_name=protein_name,
        load_path=data_path,
        cfg=cfg,
        seq=seq,
        use_mpnn=True,
    )

    # 6.training
    print("iterations considered\n", iteration)
    predicts_sorted_test = gp_optim(
        iter_train=iteration['iteration'].unique().tolist(), iter_test=None,
        embeddings_pd=embeddings, labels_pd=labels, aux_pd=aux, gp_inputs=gp_inputs, cfg=cfg,
        measured_var='activity', final_round=num_mutants, total_iteration=1, current_iteration=1, device=device, use_TS=False, is_ALDE=False, experimental=True)
    iteration_new_ids = predicts_sorted_test.sort_values(by='prediction', ascending=False).head(num_mutants).index.tolist()

    # 7.print results
    print(f"\nTested variants in this round: {len(iteration)}")
    print(iteration.variant.tolist())
    print(f"\nSelected variants in this round: {num_mutants}")
    print(predicts_sorted_test.head(num_mutants))
    
    # 8.Save results if an output_dir is provided
    if output_dir is not None:
        output_dir = os.path.join(output_dir, f'{protein_name}{postfix}', round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, 'iteration.csv'))
        predicts_sorted_test.to_csv(os.path.join(output_dir, f'this_round_predictions_topn.csv'))
        predicts_sorted_test.loc[iteration_new_ids, :].to_csv(os.path.join(output_dir, f'selected_variants_topn.csv'))

        print(f"\nData saved to {output_dir}")
    return None


if __name__ == "__main__":
    output_table = directed_evolution_exp_gp_multi(
        protein_name='RsCas12f',
        round_name='round5',
        work_dir_base='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/preprocess/RsCas12f',
        embeddings_base_path='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/preprocess/RsCas12f/plm',
        embeddings_file_names=['RsCas12f_multimut_esm2_t33_650M_UR50D.csv'],
        round_base_path='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/round',
        round_file_names=['RsCas12f_Round1.xlsx', 'RsCas12f_Round2.xlsx', 'RsCas12f_Round3.xlsx', 'RsCas12f_Round4.xlsx'],
        wt_fasta_path='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/seq/RsCas12f.fa',
        num_mutants=20,
        device='cuda:2',
        output_dir='exp/output'
    )
    
