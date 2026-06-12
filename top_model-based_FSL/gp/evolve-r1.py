import os
import random
import time
import pandas as pd
import numpy as np
import gc
import sys
from typing import List, Dict, Any, Optional, Tuple
import torch
from evolvepro.src.data import load_dms_data, load_experimental_embeddings, load_experimental_data, create_iteration_dataframes, load_config, DictToObject, get_aux_data
from gp.data import get_gp_inputs
from evolvepro.src.utils import pca_embeddings
from gp.model import first_round, top_layer, load_model, zero_shot_predicting
from Bio import SeqIO
from transformers import EsmTokenizer  
from scipy.stats import norm

# Function to run the directed evolution simulation
def directed_evolution_simulation(
    dataset_name: str,
    embeddings_path: str,
    labels_path: str,
    num_simulations: int, 
    num_iterations: int, 
    embeddings_file_type: str,
    measured_var: str,
    embeddings_type_pt: Optional[str] = None,
    num_mutants_per_round: int = 20, 
    model_name: str = 'esm2_t33_650M_UR50D',
    device: str = 'cuda:0',
    learning_strategy: str = 'topn', 
    first_round_strategy: str = 'random',
    final_round: int = 20,
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

    # Load dataset
    embeddings, labels_true = load_dms_data(dataset_name, model_name, embeddings_path, labels_path, embeddings_file_type, embeddings_type_pt)

    # Load aux data for GP
    aux = get_aux_data(
        labels_pd=labels_true, 
        load_path=labels_path,
        dataset_name=dataset_name,
    )
    #print(f'emb:\n{embeddings}\naux:\n{aux}\nlabels:\n{labels}')

    # prepare GP inputs
    wt_fasta_path = f'data/dms/wt_fasta/{dataset_name}_WT.fasta'
    seq = str(SeqIO.read(wt_fasta_path, "fasta").seq)
    gp_inputs = get_gp_inputs(
        dataset_name=dataset_name,
        load_path=labels_path,
        cfg=cfg,
        seq=seq,
        use_mpnn=True,
    )

    # Initialize the output list of metrics
    output_list = []
    model_path =f'/data1/users/weig03/data/pre_train_model/ESM/model/{model_name}'
    print('Loading finished!')
    for i in range(1, num_simulations + 1):

        # Initialize the variables
        iteration_old = None
        num_mutants_per_round_list = []
        first_round_strategy_list = []
        measured_var_list = []
        learning_strategy_list = []
        regression_type_list = []
        simulation_list =[]
        round_list = []
        median_activity_scaled_list = []
        top_variant_list = []
        top_final_round_variants_list = []
        top_activity_scaled_list = []
        spearman_corr_list = []
        activity_binary_percentage_list = []
        ndcg_list = []
        topk_pr_list = []
        
        # Initialize the list of variants for each round
        this_round_variants_list = []
        next_round_variants_list = []

        j = 0    
        while j <= num_iterations:
            # Perform mutant selection for the first round
            if j == 0:
                print(f'======================Simulation {i}======================')
                print(f'======================Round {j}======================')
                
                model, tokenizer = load_model(model_path)
                if dataset_name != 'markin':
                    labels, labels_entropy = zero_shot_predicting(model.to(device), tokenizer, seq, device, labels_true=labels_true, is_esm=True)
                if dataset_name == 'markin':
                    label_pred = pd.read_csv('data/dms/zeroshot/markin_zeroshot.csv')
                    label_pred.set_index('mutant', inplace=True)
                    labels = labels_true.merge(
                        label_pred, 
                        left_on='variant',   # 左表连接列
                        right_on='mutant',   # 右表连接列
                        how='left'           # 左连接模式
                    )
                    labels_entropy = pd.read_csv('data/dms/zeroshot/markin_zeroshot_entropy.csv')
                    print(labels_entropy)
                #print(f'labels:\n{labels}\nlabels_entropy:\n{labels_entropy}')
                labels_new, iteration_new, this_round_variants = first_round(
                    labels=labels, 
                    labels_entropy=labels_entropy,
                    num_mutants_per_round=num_mutants_per_round, 
                    first_round_strategy=first_round_strategy, 
                    random_seed=i
                )
                #print(f'labels_new:\n{labels_new}\niteration_new:\n{iteration_new}')

                model, tokenizer = None, None
                del model, tokenizer
                torch.cuda.empty_cache()
                gc.collect()
                # Append the results to the output list
                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                measured_var_list.append(measured_var)
                learning_strategy_list.append(learning_strategy)
                regression_type_list.append('GP_sequence')               
                simulation_list.append(i)
                round_list.append(j)
                # Append None values for the metrics for the first round
                median_activity_scaled_list.append("None")
                top_activity_scaled_list.append("None")
                top_variant_list.append("None")
                top_final_round_variants_list.append("None")
                activity_binary_percentage_list.append("None")
                spearman_corr_list.append("None")
                ndcg_list.append("None")
                topk_pr_list.append("None")
                # Append the variants for the first round, round 0 will have None
                this_round_variants_list.append("None")
                next_round_variants_list.append(",".join(this_round_variants))

                j += 1

            else:
                # Perform mutant selection for the subsequent rounds
                print(f'======================Round {j}======================')
                iteration_old = iteration_new
                print("iterations considered", iteration_old)
                if learning_strategy == 'random' or learning_strategy == 'topn' or learning_strategy == 'ts':
                    # Perform top_layer analysis
                    median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, ndcg, topk_pr, df_test_new, this_round_variants, candidate_TS = top_layer(
                        iter_train=iteration_old['iteration'].unique().tolist(), iter_test=None,
                        embeddings_pd=embeddings, labels_pd=labels_new, aux_pd=aux, gp_inputs=gp_inputs, cfg=cfg,
                        measured_var=measured_var, final_round=final_round, total_iteration=num_iterations, current_iteration=j, device=device, use_TS=learning_strategy=='ts')
                else:
                    ensemble_df = None
                    for rep in range(1, 6):
                        print(f'======================Rep {rep}======================')
                        # Perform top_layer analysis
                        median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, ndcg, topk_pr, df_test_new, this_round_variants, _ = top_layer(
                            iter_train=iteration_old['iteration'].unique().tolist(), iter_test=None,
                        embeddings_pd=embeddings, labels_pd=labels_new, aux_pd=aux, gp_inputs=gp_inputs, cfg=cfg,
                        measured_var=measured_var, final_round=final_round, total_iteration=num_iterations, current_iteration=j, device=device, use_TS=learning_strategy=='ts')

                        current_rep_pred = df_test_new[['variant', 'y_pred']].copy()
                        current_rep_pred.rename(columns={'y_pred': f'y_pred_rep{rep}'}, inplace=True)
                        if ensemble_df is None:
                            ensemble_df = current_rep_pred
                        else:
                            # 验证variant顺序是否一致（重要！）
                            assert ensemble_df['variant'].equals(current_rep_pred['variant']), "Order of variants is not aligned"
                            ensemble_df = pd.concat([ensemble_df, current_rep_pred[[f'y_pred_rep{rep}']]], axis=1)
                    result_df = pd.DataFrame({
                        'variant': ensemble_df['variant'],
                        'y_pred_mean': ensemble_df.filter(regex='y_pred_rep').mean(axis=1),
                        'y_pred_std': ensemble_df.filter(regex='y_pred_rep').std(axis=1),
                        'y_actual_scaled': df_test_new['y_actual_scaled']
                    })
                    '''
                    print(result_df)
                    print(result_df['y_pred_mean'].describe())
                    print(result_df['y_pred_std'].describe())
                    print(result_df['y_actual_scaled'].describe())
                    sys.exit()   
                    '''               
                # Perform mutant selection for the next round based on the results of the current round
                if learning_strategy == 'random':
                    iteration_new_ids = random.sample(list(df_test_new.variant), num_mutants_per_round)
                elif learning_strategy == 'topn':
                    iteration_new_ids = df_test_new.sort_values(by='y_pred', ascending=False).head(num_mutants_per_round).variant
                elif learning_strategy == 'ucb':
                    adaptive_beta = 1-(j / num_iterations)
                    result_df['UCB'] = result_df['y_pred_mean'] + 4 * adaptive_beta * result_df['y_pred_std']
                    iteration_new_ids = result_df.sort_values(by='UCB', ascending=False).head(num_mutants_per_round).variant
                elif learning_strategy == 'ts':
                    iteration_new_ids = df_test_new.iloc[candidate_TS]['variant'].tolist()
                    assert len(iteration_new_ids) == num_mutants_per_round, f"Expected {num_mutants_per_round} mutants, got {len(iteration_new_ids)} when using Thompson Sampling!"
                    #print(f'Thompson Sampling:\n{iteration_new_ids}')
                    #print(df_test_new.sort_values(by='y_pred', ascending=False).head(num_mutants_per_round).variant)
                    #print(f'Topk:\n{df_test_new.sort_values(by='y_pred', ascending=False).head(num_mutants_per_round).variant}')
                elif learning_strategy == 'ei':
                    #adaptive_xi = 0.01 * (1-(j / num_iterations))
                    adaptive_xi = 0.01
                    y_best = 1.0
                    mu = result_df['y_pred_mean'].values     # (n,)
                    sigma = result_df['y_pred_std'].values      # (n,)
                    
                    Z = (mu - y_best - adaptive_xi) / np.maximum(sigma, 1e-12)
                    ei_vals = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
                    result_df['EI'] = ei_vals
                    iteration_new_ids = result_df.sort_values(by='EI', ascending=False).head(num_mutants_per_round).variant
                    #print(f'EI:\n{iteration_new_ids}')
                    #print(df_test_new.sort_values(by='y_pred', ascending=False).head(num_mutants_per_round).variant)
                #else:
                #    raise ValueError(f"Unknown learning strategy: {learning_strategy}")

                iteration_new = pd.DataFrame({'variant': iteration_new_ids, 'iteration': j})
                iteration_new = pd.concat([iteration_new, iteration_old], ignore_index=True)
                labels_new = pd.merge(labels, iteration_new, on='variant', how='left')

                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                measured_var_list.append(measured_var)
                learning_strategy_list.append(learning_strategy)
                regression_type_list.append('GP_sequence')
                simulation_list.append(i)
                round_list.append(j)
                median_activity_scaled_list.append(median_activity_scaled)
                top_activity_scaled_list.append(top_activity_scaled)
                top_variant_list.append(top_variant)
                top_final_round_variants_list.append(top_final_round_variants)
                activity_binary_percentage_list.append(activity_binary_percentage)
                spearman_corr_list.append(spearman_corr)
                ndcg_list.append(ndcg)
                topk_pr_list.append(topk_pr)

                this_round_variants_list.append(",".join(iteration_old.variant))
                next_round_variants_list.append(",".join(iteration_new_ids))

                j += 1

            df_metrics = pd.DataFrame({
                'simulation_num': simulation_list, 
                'round_num': round_list, 
                'num_mutants_per_round': num_mutants_per_round_list, 
                'first_round_strategy': first_round_strategy_list, 
                'measured_var': measured_var_list, 
                'learning_strategy': learning_strategy_list, 
                'regression_type': regression_type_list,
                "spearman_corr": spearman_corr_list,
                "ndcg": ndcg_list,
                "topk_recall": topk_pr_list,
                'median_activity_scaled': median_activity_scaled_list, 
                'top_activity_scaled': top_activity_scaled_list, 
                'activity_binary_percentage': activity_binary_percentage_list, 
                "top_variant": top_variant_list, 
                "top_final_round_variants": top_final_round_variants_list, 
                "this_round_variants": this_round_variants_list, 
                "next_round_variants": next_round_variants_list
            })

        output_list.append(df_metrics)


    output_table = pd.concat(output_list)
    return output_table

# Function to run the experiment with different combinations of parameters 
def grid_search(
    dataset_name: str,
    experiment_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    num_simulations: int,
    num_iterations: List[int],
    measured_var: List[str],
    learning_strategies: List[str],
    num_mutants_per_round: List[int],
    num_final_round_mutants: int,
    first_round_strategies: List[str],
    embedding_types: List[str],
    pca_components: List[int],
    embeddings_file_type: str,
    output_dir: str,
    embeddings_type_pt: Optional[str] = None) -> None:

    """
    Run the experiment with different combinations of parameters.

    Args:
    dataset_name (str): Name of the dataset.
    experiment_name (str): Name of the experiment.
    model_name (str): Name of the model.
    embeddings_path (str): Path to the embeddings file.
    labels_path (str): Path to the labels file.
    num_simulations (int): Number of simulations to run.
    num_iterations (List[int]): List of number of iterations to run.
    measured_var (List[str]): List of measured variables.
    learning_strategies (List[str]): List of learning strategies.
    num_mutants_per_round (List[int]): List of number of mutants to select per round.
    num_final_round_mutants (int): Number of final round mutants.
    first_round_strategies (List[str]): List of first round strategies.
    embedding_types (List[str]): List of embedding types.
    pca_components (List[int]): List of PCA components.
    regression_types (List[str]): List of regression types.
    embeddings_file_type (str): Type of embeddings file.
    output_dir (str): Directory to save output files.
    embeddings_type_pt (str): Type of embeddings file (PyTorch).

    Returns:
    None
    """
    #load top_layer config
    cfg = load_config('gp/config.yaml')
    cfg = DictToObject(cfg)

    # Load dataset
    embeddings, labels = load_dms_data(dataset_name, model_name, embeddings_path, labels_path, embeddings_file_type, embeddings_type_pt)
    # Load aux data for GP
    aux = get_aux_data(
        labels_pd=labels, 
        load_path=labels_path,
        dataset_name=dataset_name,
    )
    #print(f'emb:\n{embeddings}\naux:\n{aux}\nlabels:\n{labels}')

    # prepare GP inputs
    gp_inputs = get_gp_inputs(
        dataset_name=dataset_name,
        load_path=labels_path,
        cfg=cfg
    )
    
    if embeddings is None or labels is None:
        print("Failed to load data. Exiting.")
        return

    # Generate pca components only if pca_components is not None
    if pca_components is not None:
        embeddings_pca = {
            f'embeddings_pca_{n}': pca_embeddings(embeddings, n_components=n)
            for n in pca_components
        }
        embeddings_list = {
            'embeddings': embeddings,
            **embeddings_pca
        }
    else:
        embeddings_list = {
            'embeddings': embeddings
        }

    # save the labels in a list
    output_results = {}

    # Initialize the total combinations count
    total_combinations = 0 

    # Print the total number of combinations
    for strategy in learning_strategies:
        for var in measured_var:
            for iterations in num_iterations:
                for mutants_per_round in num_mutants_per_round:
                    for embedding_type in embedding_types:
                        for first_round_strategy in first_round_strategies:
                            total_combinations += 1

    # Print the corrected total_combinations count
    print(f"Total combinations: {total_combinations}")

    # Initialize the combination count
    output_list = []
    combination_count = 0

    start_time = time.time()

    for strategy in learning_strategies:
        output_results[strategy] = {}
        for var in measured_var:
            output_results[strategy][var] = {}
            for iterations in num_iterations:
                output_results[strategy][var][iterations] = {}
                for mutants_per_round in num_mutants_per_round:
                    output_results[strategy][var][iterations][mutants_per_round] = {}
                    for embedding_type in embedding_types:
                        output_results[strategy][var][iterations][mutants_per_round][embedding_type] = {}
                        for first_round_strategy in first_round_strategies:
                            output_results[strategy][var][iterations][mutants_per_round][embedding_type][first_round_strategy] = {}
                            combination_count += 1
                            # print overall progress

                            # run simulations for current combination of parameters
                            output_table = directed_evolution_simulation(
                                labels=labels,
                                embeddings=embeddings_list[embedding_type],
                                aux=aux,
                                gp_inputs=gp_inputs,
                                cfg=cfg,
                                num_simulations=num_simulations,
                                num_iterations=iterations,
                                num_mutants_per_round=mutants_per_round,
                                measured_var=var,
                                learning_strategy=strategy,
                                final_round=num_final_round_mutants,
                                first_round_strategy=first_round_strategy,
                                embedding_type=embedding_type
                            )
                            print(
                                f"Progress: {combination_count}/{total_combinations} "
                                f"({(combination_count/total_combinations)*100:.2f}%)"
                            )

                            output_list.append(output_table)


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Total execution time: {execution_time:.2f} seconds")

    #concat the outputlist into a dataframe
    df_results = pd.concat(output_list)

    # make the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # save the dataframe to a csv file using the dataset_name
    if embeddings_type_pt == None:
        df_results.to_csv(f"{output_dir}/{dataset_name}_{model_name}_{experiment_name}.csv", index=False)
    else:
        df_results.to_csv(f"{output_dir}/{dataset_name}_{model_name}_{experiment_name}_{embeddings_type_pt}.csv", index=False)

def evolve_experimental(
    protein_name : str,
    round_name : str,
    embeddings_base_path : str,
    embeddings_file_name : str,
    round_base_path : str,
    round_file_names : List[str],
    wt_fasta_path : str,
    rename_WT : bool = False,
    number_of_variants : int = 12,
    output_dir : str = '/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/exp_results/'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 

    """
    Perform one round of directed evolution for a protein.

    Args:
    protein_name (str): Name of the protein.
    round_name (str): Name of the current round (e.g., 'Round1').
    embeddings_base_path (str): Base path for embeddings file.
    embeddings_file_name (str): Name of the embeddings file.
    round_base_path (str): Base path for round data files.
    round_file_names (list): List of round file names.
    wt_fasta_path (str): Path to the wild-type FASTA file.
    rename_WT (bool): Whether to rename the wild-type.
    number_of_variants (int): Number of top variants to display.
    output_dir (str): Directory to save output files.

    Returns:
    tuple: (this_round_variants, df_test, df_sorted_all)
    """
    
    print(f"Processing {protein_name} - {round_name}")
    
    # Load embeddings
    embeddings = load_experimental_embeddings(embeddings_base_path, embeddings_file_name, rename_WT)
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # Load experimental data
    all_experimental_data = []
    for round_file_name in round_file_names:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    
    # Create iteration dataframes
    iteration, labels = create_iteration_dataframes(all_experimental_data, embeddings.index)
    print(f"iteration shape: {iteration.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Perform top layer analysis
    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration['iteration'].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var='activity',
        regression_type='randomforest',
        experimental=True
    )
    
    # Print results
    print(f"\nTested variants in this round: {len(this_round_variants)}")
    print(this_round_variants)
    print(f"\nTop {number_of_variants} variants predicted by the model:")
    print(df_test.sort_values(by=['y_pred'], ascending=False).head(number_of_variants))
    
    # Save results if an output_dir is provided
    if output_dir is not None:
        output_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, 'iteration.csv'))
        this_round_variants.to_csv(os.path.join(output_dir, 'this_round_variants.csv'))
        df_test = df_test.sort_values(by=['y_pred'], ascending=False)
        df_test.to_csv(os.path.join(output_dir, 'df_test.csv'))
        df_sorted_all.to_csv(os.path.join(output_dir, 'df_sorted_all.csv'))
        print(f"\nData saved to {output_dir}")
    
    return this_round_variants, df_test, df_sorted_all

def evolve_experimental_multi(
    protein_name: str,
    round_name: str,
    embeddings_base_path: str,
    embeddings_file_names: List[str],
    round_base_path: str,
    round_file_names_single: List[str],
    round_file_names_multi: List[str],
    wt_fasta_path: str,
    rename_WT: bool = False,
    number_of_variants: int = 12,
    output_dir: str = '/orcd/archive/abugoot/001/Projects/Matteo/Github/EvolvePro/output/exp_results/'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform one round of directed evolution for a protein with multi-mutant support.
    """
    print(f"Processing {protein_name} - {round_name}")
    
    # Load and concatenate multiple embedding files
    embeddings_list = []
    for i, file_name in enumerate(embeddings_file_names):
        embedding = load_experimental_embeddings(embeddings_base_path, file_name, rename_WT)
        if i > 0:  # If not the first file
            embedding = embedding[~embedding.index.isin(['WT', 'WT Wild-type sequence'])]
        embeddings_list.append(embedding)
    
    embeddings = pd.concat(embeddings_list)
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # Load experimental data
    all_experimental_data = []
    for round_file_name in round_file_names_single:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path, single_mutant=True)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")

    for round_file_name in round_file_names_multi:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path, single_mutant=False)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    
    # Create iteration dataframes
    iteration, labels = create_iteration_dataframes(all_experimental_data, embeddings.index)
    
    # Perform top layer analysis
    this_round_variants, df_test, df_sorted_all = top_layer(
        iter_train=iteration['iteration'].unique().tolist(),
        iter_test=None,
        embeddings_pd=embeddings,
        labels_pd=labels,
        measured_var='activity',
        regression_type='randomforest',
        experimental=True
    )
    
    # Print results
    print(f"\nTested variants in this round: {len(this_round_variants)}")
    print(this_round_variants)
    print(f"\nTop {number_of_variants} variants predicted by the model:")
    print(df_test.sort_values(by=['y_pred'], ascending=False).head(number_of_variants)[['variant', 'y_pred']])
    
    # Save results if an output_dir is provided
    if output_dir is not None:
        output_dir = os.path.join(output_dir, protein_name, round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, 'iteration.csv'))
        this_round_variants.to_csv(os.path.join(output_dir, 'this_round_variants.csv'))
        df_test = df_test.sort_values(by=['y_pred'], ascending=False)
        df_test.to_csv(os.path.join(output_dir, 'df_test.csv'))
        df_sorted_all.to_csv(os.path.join(output_dir, 'df_sorted_all.csv'))
        print(f"\nData saved to {output_dir}")
    
    return this_round_variants, df_test, df_sorted_all

if __name__ == "__main__":
    datasets = ["lee", "stiffler",  "cas12f", "brenan", "giacomelli", "zikv_E", "markin","haddox","doud", "kelsic", "cov2_S", "jones"]
    first_round_strategys = ['random', 'topk', 'uncertainty']
    learning_strategys = ['topn', 'ei', 'ucb', 'ts', 'random']
    
    output_list = []
    for first_round_strategy in first_round_strategys:
        for learning_strategy in learning_strategys:
            for dataset in datasets:
                if os.path.exists(f'./output/dms_results_gp/{first_round_strategy}_{learning_strategy}_{dataset}.csv'):
                    print(f'{first_round_strategy}_{learning_strategy}_{dataset} has been done, skipping!')
                    output_table = pd.read_csv(f'./output/dms_results_gp/{first_round_strategy}_{learning_strategy}_{dataset}.csv')
                else:
                    output_table = directed_evolution_simulation(
                        dataset_name=dataset,
                        embeddings_path='./output/plm/esm',
                        labels_path='./output/dms',
                        embeddings_file_type='csv',
                        num_simulations=10,
                        num_iterations=5,
                        num_mutants_per_round=20,
                        model_name='esm2_t33_650M_UR50D',
                        device='cuda:0',
                        learning_strategy=learning_strategy,
                        first_round_strategy=first_round_strategy,
                        final_round=20, 
                        measured_var="activity"
                        )
                    output_table.to_csv(f"./output/dms_results_gp/{first_round_strategy}_{learning_strategy}_{dataset}.csv", index=False)
                output_list.append(output_table)
