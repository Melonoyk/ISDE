import os
import random
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import time
import math
import itertools
import sys
import gc
import re
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple
from fsfp import config
from active_learning.utils import config as al_config
from active_learning.utils.data import foldseek_struc_vocab, make_mutant_library, load_experimental_data, add_missing_variants, has_duplicates
from fsfp.pipeline import Pipeline
from fsfp.trainer import MetaRankingTrainer, ContrastiveTrainer, MetaContrastiveTrainer
from fsfp.dataset.base import MutantSequenceData, MetaRankingSequenceData
from fsfp.utils.data import make_dir, split_data
from fsfp.utils.score import metrics, calc_ndcg, calc_toprecall
from fsfp.dataset.saprot import SaProtMutantData, saprot_zero_shot

def load_model(model):
    model_name = config.model_dir[model]
    model = EsmForMaskedLM.from_pretrained(model_name)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    return model, tokenizer

def zero_shot_predicting(model, tokenizer, seq, device, labels_true=None, is_esm=True):
    if is_esm:
        mutant_library = make_mutant_library(seq)
    else:
        pure_seq = ''.join([seq[2*i] for i in range(len(seq)//2)])
        mutant_library = make_mutant_library(pure_seq)  
    if is_esm:
        wt_seq = tokenizer(seq, return_tensors='pt').to(device)
    else:
        wt_seq = tokenizer(seq, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = model(**wt_seq).logits.squeeze(0)
        log_prob = torch.log_softmax(logits, dim=-1)
    
    #calculate mutant score
    sequence_library = []
    for mutant in tqdm(mutant_library, desc='zero-shot predicting...'):
        mutant_score = 0
        wt, pos, mut = mutant
        if is_esm:
            assert seq[pos-1] == wt, f'error in making mutant library: {mutant} but {seq[pos-1]} != {wt}'
            wt_aa = tokenizer(wt, add_special_tokens=False)['input_ids']
            mt_aa = tokenizer(mut, add_special_tokens=False)['input_ids']
            mutant_score = log_prob[pos, mt_aa] - log_prob[pos, wt_aa]
        else:
            assert seq[2*(pos-1)] == wt, f'error in making mutant library: {mutant} but {seq[2*(pos-1):2*pos]} != {wt}'
            vocab = tokenizer.get_vocab()
            wt_aa = vocab[wt + foldseek_struc_vocab[0]]
            mt_aa = vocab[mut + foldseek_struc_vocab[0]]
            wt_prob = log_prob[pos, wt_aa: wt_aa + len(foldseek_struc_vocab)].mean()
            mt_prob = log_prob[pos, mt_aa: mt_aa + len(foldseek_struc_vocab)].mean()
            mutant_score += (mt_prob - wt_prob)

        mutant_save = ''.join([wt, str(pos), mut])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
    labels = pd.DataFrame(sequence_library, columns=['mutant', 'prediction'])

    labels.set_index('mutant', inplace=True)
    labels_combined = labels_true.join(labels, how='left')
    labels_combined.reset_index(inplace=True)
    if not is_esm:
        return labels_combined, None
    else:
        # Calculate entropy-based labels (modified logic)
        prob = torch.softmax(logits, dim=-1)
        entropies = {site: -torch.sum(prob[site] * torch.log(prob[site] + 1e-10)).item() 
                for site in range(len(seq))}
        
        # Generate entropy-based selection using aligned labels
        selected_mutations_entropy = []
        for site in sorted(entropies, key=entropies.get, reverse=True):
            # Filter mutations present in labels_true (if provided)
            site_mutations = labels_combined[labels_combined['mutant'].str[1:-1].astype(int, errors='ignore') == (site + 1)]

            if not site_mutations.empty:
                best_mutant = site_mutations.loc[site_mutations['prediction'].idxmax()]
                selected_mutations_entropy.append((best_mutant['mutant'], best_mutant['prediction']))
        
        labels_entropy = pd.DataFrame(selected_mutations_entropy, columns=['mutant', 'prediction'])

    return labels_combined, labels_entropy

def uncertainty_selection(predicts, logits):
    #calculate entropy of each position
    prob = torch.softmax(logits, dim=-1)
    entropies = {}
    for site in range(1, logits.size(0)-1):
        aa_probs = prob[site]
        entropy = -torch.sum(aa_probs * torch.log(aa_probs + 1e-10)).item()
        entropies[site] = entropy
    sorted_sites = sorted(entropies, key=entropies.get, reverse=True)
    #print(f'len(sorted_sites): {len(sorted_sites)}')
    selected_mutations_entropy = []
    for site in sorted_sites:
        site_mutations = predicts[predicts.index.str[1:-1].astype(int) == site]
        if site_mutations.empty:  # 检查 DataFrame 是否为空
            continue
        best_mutant = site_mutations['prediction'].idxmax()
        selected_mutations_entropy.append(best_mutant)
            
    return selected_mutations_entropy

def first_round(labels, labels_entropy=None, num_mutants_per_round=16, first_round_strategy='random',random_seed=None):
    if random_seed is not None:
            np.random.seed(random_seed)
    print("Starting labels length:", len(labels))
    variants = labels.mutant
    # Perform random first round search strategy
    if first_round_strategy == 'random':
        random_mutants = np.random.choice(variants, size=num_mutants_per_round, replace=False)
        iteration_zero_ids = random_mutants

    elif first_round_strategy == 'topk':
        top_mutants_list = labels.sort_values(by='prediction', ascending=False).head(num_mutants_per_round)['mutant'].tolist()
        iteration_zero_ids = top_mutants_list
        
    elif first_round_strategy == 'uncertainty':
        if labels_entropy is None:
            raise ValueError("labels_entropy must be provided for uncertainty strategy, which may not provide when using Saprot!")
        top_uncertain_mutant_list = labels_entropy.head(num_mutants_per_round)['mutant'].tolist()
        iteration_zero_ids = top_uncertain_mutant_list
        
    else:
        print("Invalid first round search strategy.")
        return None, None

    # Create DataFrame for the first round
    iteration_zero = pd.DataFrame({'variant': iteration_zero_ids, 'iteration': 0})
    #WT = pd.DataFrame({'variant': 'WT', 'iteration': 0}, index=[0])
    #iteration_zero = pd.concat([iteration_zero, WT], ignore_index=True)
    this_round_variants = iteration_zero.variant

    # labels_zero为一个包含全部DMS的Dataframe，除了含有标签外还有一个列储存着迭代信息
    # iteration_zero为一个dataframe,第0轮所有的突变以及迭代次数
    # this_round_variants为该轮突变
    return iteration_zero, this_round_variants

class MTLPipeline(Pipeline):
    def __init__(self, al_config, current_iteration, current_simulation, protein, seed, device, model, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None):
        al_config.train_size = current_iteration* 20
        al_config.current_iteration = current_iteration
        al_config.current_simulation = current_simulation
        al_config.protein = protein
        al_config.seed = seed
        al_config.model = model
        super().__init__(al_config, data_constructor, lora_modules, score_fn)
        self.device=device

    
    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        if prefix == 'meta':
            save_dir = 'ALDE/{}/{}/{}/{}/r{}{}{}{}{}{}{}{}{}'.format(
                config.pred_dir if prediction else config.ckpt_dir,
                prefix,
                args.model,
                protein_name,
                args.lora_r,
                f'_ts{args.train_size}_cv{args.cross_validation}' if not (args.augment and prefix == 'finetune') else '',
                f'_{args.retr_metric}_mt{args.meta_tasks}' if 'meta' in prefix else '',
                '_' + '-'.join(args.augment) if args.augment else '',
                '_regr' if args.list_size == 1 else '',
                '_ms' if args.n_sites != [1] else '',
                args.save_postfix,
                '_contrast' if args.use_contrast else '',
                '_msa' if args.use_msa and prediction else '')
        else:
            save_dir = 'ALDE/{}/{}/{}/{}/{}/r{}{}{}{}{}{}{}{}{}'.format(
                f'simulation{args.current_simulation}',
                config.pred_dir if prediction else config.ckpt_dir,
                prefix,
                args.model,
                protein_name,
                args.lora_r,
                f'_ts{args.train_size}_cv{args.cross_validation}' if not (args.augment and prefix == 'finetune') else '',
                f'_{args.retr_metric}_mt{args.meta_tasks}' if 'meta' in prefix else '',
                '_' + '-'.join(args.augment) if args.augment else '',
                '_regr' if args.list_size == 1 else '',
                '_ms' if args.n_sites != [1] else '',
                args.save_postfix,
                '_contrast' if args.use_contrast else '',
                '_msa' if args.use_msa and prediction else '')
        return save_dir
    
    def meta_single(self, train, eval_train, eval_test=None, use_contrast=False):
        args = self.args
        args.mode = 'meta'
        model, tokenizer = self.get_base_model()
        if args.lora_r > 0:
            lora_config = LoraConfig(r=args.lora_r,
                                lora_alpha=args.lora_r,
                                target_modules=self.lora_modules,
                                lora_dropout=0.1,
                                bias='none')
            model = get_peft_model(model, lora_config)
        
        train_splits = [split_data(protein, 0.5, True) for protein in train]
        train_data = MetaRankingSequenceData(train_splits, tokenizer,
                                                adapt_batch_size=args.meta_train_batch,
                                                eval_batch_size=args.meta_eval_batch,
                                                adapt_steps=args.adapt_steps,
                                                mask=args.mask,
                                                list_size=args.list_size,
                                                training=True,
                                                constructor=self.data_constructor,
                                                device=self.device)

        train_iter = DataLoader(train_data,
                                batch_size=args.train_batch,
                                shuffle=True,
                                collate_fn=train_data.collate)
        
        save_dir = self.get_save_dir('meta', eval_train['name'])
        if os.path.exists(save_dir + '/logs.pkl'):
            print(f'{save_dir}/logs.pkl already exist, skipping!')
            report = torch.load(save_dir + '/logs.pkl')
            return report
        if not use_contrast:
            trainer = MetaRankingTrainer(model.to(self.device),
                                        optimizer=args.optimizer,
                                        lr=args.learning_rate,
                                        epochs=args.epochs,
                                        max_grad_norm=args.max_grad_norm,
                                        score_fn=self.score_fn,
                                        adapt_lr=args.adapt_lr,
                                        eval_metric=args.eval_metric,
                                        log_metrics=metrics,
                                        save_dir=save_dir,
                                        patience=args.patience)
        else:
            trainer = MetaContrastiveTrainer(model.to(self.device),
                                        #model_reg=model_reg.to(self.device),
                                        optimizer=args.optimizer,
                                        lr=args.learning_rate,
                                        epochs=args.epochs,
                                        max_grad_norm=args.max_grad_norm,
                                        score_fn=self.score_fn,
                                        adapt_lr=args.adapt_lr,
                                        eval_metric=args.eval_metric,
                                        log_metrics=metrics,
                                        save_dir=save_dir,
                                        patience=args.patience)
        
        report = {}
        if args.cross_validation > 0:
            if args.cross_validation == 1:
                eval_splits = [(eval_train, eval_test)]
            else:
                cv_size = self.get_cv_size(eval_train)
                eval_splits = [split_data(eval_train, cv_size, True) for _ in range(args.cross_validation)]
            eval_data = MetaRankingSequenceData(eval_splits, tokenizer,
                                                    adapt_batch_size=args.meta_train_batch,
                                                    eval_batch_size=args.eval_batch,
                                                    adapt_steps=args.adapt_steps,
                                                    mask=args.mask,
                                                    list_size=args.list_size,
                                                    training=False,
                                                    constructor=self.data_constructor,
                                                    device=self.device)
        
            eval_iter = DataLoader(eval_data,
                                   batch_size=1,
                                   collate_fn=eval_data.collate)
            _, report['baseline'] = trainer.evaluate_epoch(eval_iter)
        else:
            eval_iter = None
        logs = trainer(train_iter, eval_iter)
        report.update(logs)

        report['best_epoch'] = trainer.best_epoch
        torch.save(report, save_dir + '/logs.pkl')
        return report
    
    def finetune_single_cv(self, train, test=None, use_contrast=False):
        args = self.args
        args.mode = 'meta-transfer'
        save_dir = self.get_save_dir('meta-transfer', train['name'])
        
        #if os.path.exists(save_dir + '/logs.pkl'):
        #    print(f'{save_dir}/logs.pkl already exist, skipping!')
        #    report = torch.load(save_dir + '/logs.pkl')
        #    return report
        
        cv_size = self.get_cv_size(train)
        splits = [split_data(train, cv_size, True) for _ in range(args.cross_validation)]
        epochs = args.epochs
        for i, (cv_train, cv_valid) in enumerate(splits):
            print(f'======================Cross validation: Split {i + 1}======================')
            cv_report = self.finetune_single(cv_train, cv_valid, use_contrast=use_contrast)
            args.epochs = min(args.epochs, len(cv_report[args.eval_metric]))
            if i == 0:
                report = cv_report
                continue
            for key, value in cv_report['baseline'].items():
                report['baseline'][key] += value
            for key in metrics:
                for i in range(args.epochs):
                    report[key][i] += cv_report[key][i]
        for key in report['baseline'].keys():
            report['baseline'][key] /= len(splits)
        for key in metrics:
            report[key] = [value / len(splits) for value in report[key][:args.epochs]]
            if key == args.eval_metric: # find best epoch based on cv scores
                best_epoch, best_score = max(enumerate(report[key]), key=lambda x: x[1])
                best_epoch += 1
        print(f'CV-estimated best validating {args.eval_metric} reached at epoch {best_epoch}: {best_score:.3f}')
        print(f'----------------------Training on full data for {best_epoch} epochs----------------------')
        report['best_epoch'] = args.epochs = best_epoch
        logs = self.finetune_single(train, None, save_dir, use_contrast)
        report['train_loss'] = logs['train_loss']
        torch.save(report, save_dir + '/logs.pkl')
        args.epochs = epochs
        return report

    def test_single(self, train, test, use_msa=False, seq_aln_file=None):
        args = self.args
        args.mode = 'meta-transfer'
        args.test = True
        if args.epochs > 0:
            load_dir = self.get_save_dir(args.mode, test['name'])
            if args.lora_r == 0:
                model, tokenizer = self.get_base_model(load_dir)
            else:
                model, tokenizer = self.get_base_model()
                model = PeftModel.from_pretrained(model, load_dir, is_trainable=True)
        else:
            model, tokenizer = self.get_base_model()
        train_data = self.data_constructor(train, tokenizer,
                                        mask=args.mask in {'eval', 'all'},
                                        device=self.device)
        test_data = self.data_constructor(test, tokenizer,
                                          mask=args.mask in {'eval', 'all'},
                                          device=self.device)
        train_iter = DataLoader(train_data,
                              batch_size=args.eval_batch,
                              collate_fn=train_data.collate)
        test_iter = DataLoader(test_data,
                               batch_size=args.eval_batch,
                               collate_fn=test_data.collate)

        trainer = ContrastiveTrainer(model=model.to(self.device), tokenizer=tokenizer, 
                                        log_metrics=[], score_fn=self.score_fn, use_msa=use_msa, 
                                        seq_aln_file=seq_aln_file, alpha=args.alpha)
        predicts_train, _, _ = trainer.evaluate_epoch(train_iter, is_active=True)
        predicts_test, _, logits = trainer.evaluate_epoch(test_iter, is_active=True)
        predicts_train, predicts_test = predicts_train.tolist(), predicts_test.tolist()
        predicts_df_train = pd.DataFrame(predicts_train, index=train['df'].index, columns=["prediction"])
        predicts_df_train = predicts_df_train.join(train['df'][['DMS_score_scaled', 'DMS_score_bin']], how='left')
        predicts_df_test = pd.DataFrame(predicts_test, index=test['df'].index, columns=["prediction"])
        predicts_df_test = predicts_df_test.join(test['df'][['DMS_score_scaled', 'DMS_score_bin']], how='left')
        predicts = pd.Series(predicts_test, index=test['df'].index, name='prediction')
        #report, _ = group_scores(train['df'], predicts, test['df'])
        #print('======================Breakdown results======================')
        #print(report)
        
        print('Saving model predictions...')
        save_path = self.get_save_dir(args.mode, test['name'], prediction=True)
        save_path += '_base.csv' if args.epochs == 0 else '.csv'
        make_dir(save_path)
        predicts.to_csv(save_path)

        return predicts_df_train, predicts_df_test, logits

    def __call__(self, all_proteins, iteration_old, postfix=''):
        args = self.args
        proteins = self.select_datasets(all_proteins)
        database, topk = self.get_meta_database(all_proteins)
        topk = torch.load(f'{config.retr_dir}/topk_{args.model}_{args.retr_metric}_AL.pkl')
        print('Starting using meta learning...')
        for protein in proteins:
            if protein['name'] == args.protein + postfix:
                print(f'**********************Current dataset: {protein["name"]}**********************')
                train, test = self.custom_split(protein, iteration_old)
                # meta learning
                src_name = '_'.join(protein['name'].split('_')[:2])
                #根据query：src_name取出topk个proteingym数据集
                tgt_ls = topk[src_name]['tgt_names']
                if src_name == 'zikv_E':
                    tgt_ls.remove('A0A140D2T1_ZIKV')
                    tgt_ls.remove('A0A097PF60_9INFA')
                elif src_name == 'cov2_S':
                    tgt_ls.remove('cov2_S')
                elif src_name == 'A0A097PF60_9INFA':
                    tgt_ls.remove('A0A2Z5U3Z0_9INFA')
                    tgt_ls.remove('zikv_E')
                elif src_name == 'A0A2Z5U3Z0_9INFA':
                    tgt_ls.remove('A0A097PF60_9INFA')
                    tgt_ls.remove('zikv_E')
                    
                tgt_names = tgt_ls[:args.meta_tasks]
                #根据tgt_names从database中取出元任务训练集
                meta_train = [database[name] for name in tgt_names]
                meta_train[-1:] = self.augment_data(protein)
                if args.meta_tasks < 4:
                    meta_train *= 2
                #meta_train为一个列表列表里面每个元素为一个字典即被选出的topk个数据集
                #train和test则是依据split_data在这个目标蛋白数据集上划分出来的数据集
                #返回一个字典，键为lr，train_loss，评估指标，最佳表现的周期,baseline
                print('Meta learning on anxilar datasets...')
                _ = self.meta_single(meta_train, train, test, args.use_contrast)

                # meta transfer learning
                print('Meta transfer learning on target dataset...')
                _ = self.finetune_single_cv(train, test, use_contrast=args.use_contrast)

                # evaluation
                print('Evaluation on target dataset...')
                seq_aln_file = f"{args.seq_aln_dir}/{protein['name']}.a2m"
                predicts_train, predcits_test, logits = self.test_single(train=train, test=test, 
                                            use_msa=args.use_msa, seq_aln_file=seq_aln_file)
                print('Meta transfer learning finished!')
                
        return predicts_train, predcits_test, logits.squeeze(0)
    
    def custom_split(self, protein, iteration_old):
        df = protein['df']
        df['DMS_score_scaled'] = (df['DMS_score'] - df['DMS_score'].min()) / (df['DMS_score'].max() - df['DMS_score'].min())
        train, test = protein.copy(), protein.copy()
        variants = iteration_old['variant'].tolist()
        train['df'] = df.loc[variants]
        test['df'] = df.loc[df.index.difference(variants, sort=False)]

        return train, test

# Function to run the directed evolution simulation
def directed_evolution_simulation(
    protein: str,
    num_simulations: int, 
    num_iterations: int, 
    num_mutants_per_round: int = 20, 
    model_id: str = 'esm2',
    device: str = 'cuda:0',
    learning_strategy: str = 'topn', 
    mix_ratio: float = 0.75, 
    first_round_strategy: str = 'random',
    final_round: int = 20,
    postfix: str = '',
    use_top_model: bool = False,
    top_model: str = 'dkl',
    ) -> pd.DataFrame:

    """
    Run the directed evolution simulation.

    Args:
    labels (pd.DataFrame): DataFrame of labels.
    num_simulations (int): Number of simulations to run.
    num_iterations (int): Number of iterations to run.
    num_mutants_per_round (int): Number of mutants to select per round.
    measured_var (str): Measured variable.
    learning_strategy (str): Learning strategy.
    top_n (int): Number of top variants to consider.
    final_round (int): Number of final round mutants.
    first_round_strategy (str): First round strategy.
    explicit_variants (list): List of explicit variants.

    Returns:
    pd.DataFrame: DataFrame of simulation results.
    """
    print(f'Running directed evolution simulation for {protein}{postfix}...')
    # Initialize the output list of metrics
    output_list = []
    path = config.data_path.replace('.pkl', f'_AL.pkl')
    all_proteins = torch.load(path)
    wt_fasta_path = f'active_learning/data/seq/{protein}.fasta'
    if model_id == 'esm2':
        seq = str(SeqIO.read(wt_fasta_path, "fasta").seq)
    else:
        struc_seqs = pd.read_csv(config.struc_seq_path, index_col='protein')
        seq = struc_seqs.loc[protein, 'struc_sequence']
    print('Loading finished!')
    # Simulate the directed evolution process
    for i in range(1, num_simulations + 1):

        # Initialize the variables
        iteration_old = None
        num_mutants_per_round_list = []
        first_round_strategy_list = []
        learning_strategy_list = []
        simulation_list =[]
        round_list = []
        top_final_round_variants_list = []
        top_variant_list = []
        top_activity_scaled_list = []
        median_activity_scaled_list = []
        activity_binary_percentage_list = []
        spearman_corr_list = []
        ndcg_list = []
        topk_pr_list = []
        
        # Initialize the list of variants for each round
        this_round_variants_list = []
        next_round_variants_list = []

        j = 0    
        while j <= num_iterations:
            # Perform mutant selection for the first round
            if j == 0:
                # labels_new为一个包含全部DMS的Dataframe，除了含有标签外还有一个列储存着迭代信息
                # iteration_new为一个dataframe,第0轮所有的突变以及迭代次数
                # this_round_variants为该轮突变，series类
                print(f'======================Simulation {i}======================')
                print(f'======================Round {j}======================')
                model, tokenizer = load_model(model_id)
                proteins = all_proteins[protein]
                for protein_sub in proteins:
                    if protein_sub['name'] == protein + postfix:
                        labels_true = protein_sub['df']
                labels, labels_entropy = zero_shot_predicting(model.to(device), tokenizer, seq, device, labels_true=labels_true, is_esm=model_id=='esm2')
                iteration_new, this_round_variants = first_round(
                    labels=labels, 
                    labels_entropy=labels_entropy,
                    num_mutants_per_round=num_mutants_per_round, 
                    first_round_strategy=first_round_strategy, 
                    random_seed=i
                )
                model, tokenizer = None, None
                del model, tokenizer
                torch.cuda.empty_cache()
                gc.collect()
                # Append the results to the output list
                num_mutants_per_round_list.append(num_mutants_per_round)
                first_round_strategy_list.append(first_round_strategy)
                learning_strategy_list.append(learning_strategy)            
                simulation_list.append(i)
                round_list.append(j)
                top_final_round_variants_list.append("None")
                top_variant_list.append("None")
                top_activity_scaled_list.append("None")
                median_activity_scaled_list.append("None")
                activity_binary_percentage_list.append("None")
                # Append None values for the metrics for the first round
                spearman_corr_list.append("None")
                ndcg_list.append("None")
                topk_pr_list.append("None")
                # Append the variants for the first round, round 0 will have None
                this_round_variants_list.append("None")
                next_round_variants_list.append(",".join(this_round_variants))

                j += 1

            else:
                # Perform mutant selection for the subsequent rounds
                # dataframe, 两列，virants，iteration
                print(f'======================Round {j}======================')
                iteration_old = iteration_new
                print("iterations considered\n", iteration_old)
                if learning_strategy == 'random' or learning_strategy == 'topn':
                    if model_id == 'saprot':
                        mtlpipeline = MTLPipeline(
                            al_config=al_config,
                            current_iteration=j,
                            current_simulation=i,
                            protein=protein,
                            seed=i, 
                            device=device,
                            data_constructor=SaProtMutantData,
                            score_fn=saprot_zero_shot,
                            model=model_id
                        )
                    else:
                        mtlpipeline = MTLPipeline(
                            al_config=al_config,
                            current_iteration=j,
                            current_simulation=i,
                            protein=protein,
                            seed=i,
                            device=device,
                            model=model_id
                        )
                    predicts_train, predicts_test, logits = mtlpipeline(all_proteins, iteration_old, postfix=postfix)
                    # Perform mutant selection for the next round based on the results of the current round
                    predicts_all = pd.concat([predicts_train, predicts_test])
                    predicts_sorted_all = predicts_all.sort_values(by='prediction', ascending=False)
                    predicts_sorted_train = predicts_train.sort_values(by='DMS_score_scaled', ascending=False)
                else:
                    ensemble_df = None
                    for rep in range(1, 6):
                        random_seed = str(i) + str(j) + str(rep)
                        random_seed = int(random_seed)
                        print(f'======================Rep {rep}======================')
                        if model_id =='saprot':
                            mtlpipeline = MTLPipeline(
                                al_config=al_config,
                                current_iteration=j,
                                current_simulation=i,
                                protein=protein,
                                seed=random_seed,
                                device=device,
                                data_constructor=SaProtMutantData,
                                score_fn=saprot_zero_shot,
                                model=model_id
                            )
                        else:
                            mtlpipeline = MTLPipeline(
                                al_config=al_config,
                                current_iteration=j,
                                current_simulation=i,
                                protein=protein,
                                seed=random_seed,
                                device=device,
                                model=model_id
                            )
                        predicts_train, predicts_test, logits = mtlpipeline(all_proteins, iteration_old, postfix=postfix)
                        #print(f'predicts_train:\n{predicts_train}')
                        current_rep_pred = predicts_test[['prediction']].copy()
                        # 重命名
                        current_rep_pred.rename(
                            columns={'prediction': f'prediction_rep{rep}'},
                            inplace=True
                        )

                        if ensemble_df is None:
                            ensemble_df = current_rep_pred
                        else:
                            # 确保 index 对齐
                            assert ensemble_df.index.equals(current_rep_pred.index), \
                                "Indexes (mutant) are not aligned!"
                            # 拼接新一列
                            ensemble_df = pd.concat(
                                [ensemble_df, current_rep_pred],
                                axis=1
                            )
                    pred_cols = ensemble_df.filter(regex=r'^prediction_rep').columns
                    prediction_mean = ensemble_df[pred_cols].mean(axis=1)
                    prediction_std  = ensemble_df[pred_cols].std(axis=1)

                    # 从最后一次的 predicts_test 中取原始标签
                    # （如果你希望用所有 rep 中的某一次都相同，可以放在循环外，但通常最后一次没问题）
                    scaled = predicts_test['DMS_score_scaled']
                    binning = predicts_test['DMS_score_bin']
                    
                    
                    # 构造结果表，以 mutant 为索引
                    result_df = pd.DataFrame({
                        'prediction_mean':      prediction_mean,
                        'prediction_std':       prediction_std,
                        'DMS_score_scaled':     scaled,
                        'DMS_score_bin':        binning,
                    }, index=ensemble_df.index)   
                    '''
                    print(result_df)
                    print(result_df['prediction_mean'].describe())
                    print(result_df['prediction_std'].describe())
                    '''
                        
                #report: dataframe, 行为all-rest...列为spearman...
                #predicts_test: dataframe, 列为prediction,索引为mutant
                #logits: (num_mut, length, 20)
                #predicts_train, predicts_test, logits = mtlpipeline(all_proteins, iteration_old, postfix=postfix)
                # Perform mutant selection for the next round based on the results of the current round
                predicts_all = pd.concat([predicts_train, predicts_test])
                predicts_sorted_all = predicts_all.sort_values(by='prediction', ascending=False)
                predicts_sorted_train = predicts_train.sort_values(by='DMS_score_scaled', ascending=False)
                
                if learning_strategy == 'mix':
                    if model_id!='esm2':
                        print(f'Learning strategy: mix is only supported for esm2!')
                        sys.exit()
                    selected_mutation_entropy = uncertainty_selection(predicts_test, logits)
                    predicts_test.sort_values(by='prediction', ascending=False)
                    num_top = math.ceil(num_mutants_per_round * mix_ratio)
                    num_entropy = num_mutants_per_round - num_top
                    iteration_new_ids_top = predicts_test.sort_values(by='prediction', ascending=False).head(num_mutants_per_round).index.tolist()
                    iteration_new_ids_entropy = selected_mutation_entropy[:num_entropy]
                    iteration_new_ids_top_nr = [x for x in iteration_new_ids_top if x not in iteration_new_ids_entropy]
                    iteration_new_ids_top_selected = iteration_new_ids_top_nr[:num_top]
                    iteration_new_ids = iteration_new_ids_top_selected + iteration_new_ids_entropy
                elif learning_strategy == 'random':
                    np.random.seed(i) 
                    iteration_new_ids = random.sample(list(predicts_test.index), num_mutants_per_round)
                elif learning_strategy == 'topn':
                    iteration_new_ids = predicts_test.sort_values(by='prediction', ascending=False).head(num_mutants_per_round).index.tolist()
                elif learning_strategy == 'ucb':
                    adaptive_beta = 1-(j / num_iterations)
                    result_df['UCB'] = result_df['prediction_mean'] + 2 * adaptive_beta * result_df['prediction_std']
                    iteration_new_ids = result_df.sort_values(by='UCB', ascending=False).head(num_mutants_per_round).index.tolist()
                    #print(f'UCB:\n{iteration_new_ids}')
                    #print(predicts_test.sort_values(by='prediction', ascending=False).head(num_mutants_per_round).index.tolist())
                    #sys.exit()
                elif learning_strategy == 'ei':
                    #adaptive_xi = 0.01 * (1-(j / num_iterations))
                    adaptive_xi = 0.01
                    y_best = 1.0
                    mu = result_df['prediction_mean'].values     # (n,)
                    sigma = result_df['prediction_std'].values      # (n,)
                    
                    Z = (mu - y_best - adaptive_xi) / np.maximum(sigma, 1e-12)
                    ei_vals = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
                    result_df['EI'] = ei_vals
                    iteration_new_ids = result_df.sort_values(by='EI', ascending=False).head(num_mutants_per_round).index.tolist()
                iteration_new = pd.DataFrame({'variant': iteration_new_ids, 'iteration': j})
                iteration_new = pd.concat([iteration_new, iteration_old], ignore_index=True)

                #predicts_sorted_test = predicts_test.sort_values(by='prediction', ascending=False)
                top_final_round_variants = predicts_sorted_train.head(final_round).index.tolist()
                median_activity_scaled = predicts_sorted_train.loc[top_final_round_variants, 'DMS_score_scaled'].median()
                top_activity_scaled = predicts_sorted_train.loc[top_final_round_variants, 'DMS_score_scaled'].max()
                top_variant = predicts_sorted_train.loc[predicts_sorted_train['DMS_score_scaled'] == top_activity_scaled].index
                top_variant = top_variant[0]
                activity_binary_percentage = predicts_sorted_train.loc[top_final_round_variants, 'DMS_score_bin'].mean()
                
                num_mutants_per_round_list.append(num_mutants_per_round)
                learning_strategy_list.append(learning_strategy)
                first_round_strategy_list.append(first_round_strategy)
                simulation_list.append(i)
                round_list.append(j)
                top_final_round_variants_list.append(",".join(top_final_round_variants))
                top_activity_scaled_list.append(top_activity_scaled)
                top_variant_list.append(top_variant)
                median_activity_scaled_list.append(median_activity_scaled)
                activity_binary_percentage_list.append(activity_binary_percentage)
                
                spearmanr = predicts_sorted_all[['prediction', 'DMS_score_scaled']].corr(method='spearman').iloc[0, 1]
                ndcg = calc_ndcg(np.array(predicts_sorted_all['DMS_score_scaled']), np.array(predicts_sorted_all['prediction']), quantile=False, top=10)
                topk_pr = calc_toprecall(np.array(predicts_sorted_all['DMS_score_scaled']), np.array(predicts_sorted_all['prediction']), top_true=10, top_model=10)
                spearman_corr_list.append(spearmanr)
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
                'learning_strategy': learning_strategy_list, 
                "top_final_round_variants": top_final_round_variants_list,
                "top_variant": top_variant_list,
                "top_activity_scaled": top_activity_scaled_list,
                "median_activity_scaled": median_activity_scaled_list,
                "activity_binary_percentage": activity_binary_percentage_list,
                "spearman_corr": spearman_corr_list,
                "ndcg": ndcg_list,
                "topk_pr": topk_pr_list,
                "this_round_variants": this_round_variants_list, 
                "next_round_variants": next_round_variants_list
            })
            #print(df_metrics)

        output_list.append(df_metrics)


    output_table = pd.concat(output_list)
    return output_table

# Function to run the experiment with different combinations of parameters 
def grid_search(
    dataset_name: str,
    model_name: str,
    device: str,
    num_simulations: int,
    num_iterations: List[int],
    learning_strategies: List[str],
    num_mutants_per_round: List[int],
    first_round_strategies: List[str],
    mix_ratio: List[float],
    output_dir: str) -> None:

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

    
    # save the labels in a list
    output_results = {}

    # Initialize the total combinations count
    total_combinations = 0 

    # Print the total number of combinations
    for strategy in learning_strategies:
        for iterations in num_iterations:
            for mutants_per_round in num_mutants_per_round:
                for first_round_strategy in first_round_strategies:
                    total_combinations += 1

    # Print the corrected total_combinations count
    # 统计网格数
    print(f"Total combinations: {total_combinations}")

    # Initialize the combination count
    output_list = []
    combination_count = 0

    start_time = time.time()

    for strategy in learning_strategies:
        output_results[strategy] = {}
        for iterations in num_iterations:
            output_results[strategy][iterations] = {}
            for mutants_per_round in num_mutants_per_round:
                output_results[strategy][iterations][mutants_per_round] = {}
                for first_round_strategy in first_round_strategies:
                    output_results[strategy][iterations][mutants_per_round][first_round_strategy] = {}
                    combination_count += 1
                    # print overall progress

                    # run simulations for current combination of parameters
                    output_table = directed_evolution_simulation(
                        protein=dataset_name,
                        num_simulations=num_simulations,
                        num_iterations=iterations,
                        num_mutants_per_round=mutants_per_round,
                        model=model_name,
                        device=device,
                        learning_strategy=strategy,
                        mix_ratio=mix_ratio,
                        first_round_strategy=first_round_strategy,
                        final_round=20,
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
    df_results.to_csv(f"{output_dir}/{dataset_name}_{model_name}.csv", index=False)
'''
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
    #加载嵌入数据并打印形状
    #返回：pd.DataFrame: Experimental embeddings
    embeddings = load_experimental_embeddings(embeddings_base_path, embeddings_file_name, rename_WT)
    print(f"Embeddings loaded: {embeddings.shape}")
    
    # Load experimental data
    #遍历 round_file_names，使用 load_experimental_data 函数加载每个轮次的实验数据，并打印每个轮次数据的形状。
    all_experimental_data = []
    for round_file_name in round_file_names:
        #返回：pd.DataFrame: Processed experimental data.
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    
    # Create iteration dataframes
    # 创建用于迭代学习训练和测试的dataframe
    # iteration为一个含有variants，iteration两列的dataframe，用于表示已经做过实验的突变体及他的实验轮次
    # labels为一个含有variants，iteration，activity，binary，scale的包括了该蛋白饱和突变的一个dataframe，其中没实验的那些突变的相应数据用nan填充
    iteration, labels = create_iteration_dataframes(all_experimental_data, embeddings.index)
    print(f"iteration shape: {iteration.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Perform top layer analysis
    #使用top_layer函数自动依据iter_train提及的迭代轮次在DMS上划分训练测试集然后进行训练和预测
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

    # 相较于evolve_experimental，此处还要额外读入多位点的实验结果
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
'''
if __name__ == '__main__':
    proteins = [
        "P53_HUMAN", 'MK01_HUMAN', 'A0A2Z5U3Z0_9INFA', 'ENV_HV1BR', 'ADRB2_HUMAN', "BLAT_ECOLX", 
        "IF1_ECOLI", "A0A097PF60_9INFA", "cov2_S", "CS12F_SULT2", "PafA_XXXX", "zikv_E"
        ]
    postfixs = [
        "_Giacomelli_NULL_Etoposide_2018_aux", "_Brenan_2016_aux", "_Doud_2016_aux", "_Haddox_2016_aux", "_Jones_2020_aux", "_Stiffler_2015_aux",
        "_Kelsic_2016_aux", "_lee_2023_aux", "_XXXX_XXXX_aux", "_Hino_2023_aux", "_XXXX_XXXX_aux", "_XXXX_XXXX_aux"
        ]
    first_round_strategys = ['random', 'topk', 'uncertainty']
    learning_strategys = ['random', 'topn', 'ei', 'ucb']
    
    output_list = []
    for first_round_strategy in first_round_strategys:
        for learning_strategy in learning_strategys:
            for protein, postfix in zip(proteins, postfixs):
                if os.path.exists(f'./tables_alde/{first_round_strategy}_{learning_strategy}_{protein}{postfix}.csv'):
                    print(f'{first_round_strategy}_{learning_strategy}_{protein}{postfix} has been done, skipping!')
                    output_table = pd.read_csv(f'./tables_alde/{first_round_strategy}_{learning_strategy}_{protein}{postfix}.csv')
                else:
                    output_table = directed_evolution_simulation(
                        protein=protein,
                        num_simulations=10,
                        num_iterations=5,
                        num_mutants_per_round=20,
                        model_id='esm2', # 'esm2' or 'saprot' 
                        device='cuda:0',
                        learning_strategy=learning_strategy,
                        first_round_strategy=first_round_strategy,
                        final_round=20, 
                        postfix=postfix
                        )
                    output_table.to_csv(f"./tables_alde/{first_round_strategy}_{learning_strategy}_{protein}{postfix}.csv", index=False)
                output_list.append(output_table)
    output_table = pd.concat(output_list)
    output_table.to_csv(f"./tables_alde/total/all.csv", index=False)
    
    