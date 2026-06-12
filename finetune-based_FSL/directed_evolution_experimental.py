import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import EsmTokenizer, EsmForMaskedLM
from fsfp.dataset.base import ProteinSequenceData
import time
import math
import itertools
import sys
import gc
import re
from collections import defaultdict
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio import SeqIO
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from fsfp import config
from active_learning.utils import config as al_config
from active_learning.utils.data import foldseek_struc_vocab, residue_vocab, make_mutant_library, load_experimental_data, create_iteration_dataframes, add_missing_variants, has_duplicates
from fsfp.pipeline import Pipeline
from directed_evolution import MTLPipeline
from fsfp.dataset.base import MutantSequenceData
from fsfp.dataset.saprot import SaProtMutantData, saprot_zero_shot
from fsfp.retrieval import Protein2Vector
from fsfp.utils.data import split_data

# 1.auxiliary data combination
def load_gemme_data(fasta_path, score_path):
    # 步骤1：读取蛋白质序列
    protein_seq = read_fasta(fasta_path)
    num_aa = len(protein_seq)
    # 步骤2：生成突变库
    mutants = make_mutant_library(protein_seq)
    
    # 步骤3：解析分数文件
    score_data = parse_score_file(score_path, num_aa)
    
    # 步骤4：为每个突变体匹配分数
    scored_mutants = []
    for orig_aa, pos, mut_aa in mutants:
        # 获取对应位置的分数（位置-1转换为0-based索引）
        score = score_data[mut_aa][pos-1]
        mutant = orig_aa + str(pos) + mut_aa
        scored_mutants.append({
            'mutant': mutant,
            'GEMME': score
        })
    gemme_df = pd.DataFrame(scored_mutants)

    # 修改列名（保持GEMME列名不变）
    final_df = gemme_df.rename(columns={
        'mutant': 'variant',
        'DMS_score': 'activity'
    })
    
    return final_df, protein_seq

# 读取FASTA文件（简化版，无需biopython）
def read_fasta(fasta_path):
    seq = str(SeqIO.read(fasta_path, "fasta").seq)
    return seq

# 解析分数文件
def parse_score_file(score_path, num_aa):
    """返回字典格式：{氨基酸: [位置1分数, 位置2分数,...]}"""
    with open(score_path) as f:
        # 跳过首行，读取后续20行
        lines = [line.strip() for line in f.readlines()[1:21]]  
    
    score_dict = {}
    for line in lines:
        parts = line.split()
        assert len(parts) == num_aa+1, f"Error: Expected {len+1} values in line, but got {len(parts)}"
        aa = parts[0].strip('"').upper() # 统一转为大写
        scores = []
        for val in parts[1:]:  # 从第二个位置开始
            scores.append(float(val) if val != 'NA' else np.nan)
        score_dict[aa] = scores
        
    return score_dict

# 2.reconstruct mtl pipeline
class MTLPipeline_exp(MTLPipeline):
    def __init__(self, al_config, train_size, protein, seed, device, model, data_constructor=MutantSequenceData,
                 lora_modules=config.lora_modules, score_fn=None):
        al_config.train_size = train_size
        al_config.protein = protein
        al_config.seed = seed
        al_config.model = model
        Pipeline.__init__(self, al_config, data_constructor, lora_modules, score_fn)
        self.device=device

    def get_save_dir(self, prefix, protein_name, prediction=False):
        args = self.args
        if prefix == 'meta':
            save_dir = 'exp/{}/{}/{}/{}/{}/r{}{}{}{}{}{}{}{}{}'.format(
                args.protein,
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
            save_dir = 'exp/{}/{}/{}/{}/{}/r{}{}{}{}{}{}{}{}{}'.format(
                args.protein,
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

    def __call__(self, all_proteins, iteration_old, protein_tgt, postfix_topk='_AL', postfix='', use_bo=False):
        args = self.args
        
        # 0.split train and test set
        train, test = self.custom_split(protein_tgt, iteration_old)
        
        # 1.vectorize target protein
        print('Vectorize target protein...')
        vector_tgt = self.vectorize(protein_tgt)
        
        # 2.calculate similarity scores and pick meta tasks
        print('Calculate similarity scores between target protein and proteins in database...')
        path =  f'{config.retr_dir}/vectors_{args.model}{postfix_topk}.pkl'
        print(f'Auxilary datasets are loaded from: {path}')
        names_src, vector_src = torch.load(path).values()
        topk, topk_idx = self.compute_nns(vector_tgt, vector_src, k=args.meta_tasks)
        for idxs in topk_idx:
            tgt_names = [names_src[idx] for idx in idxs]
        print(f'Chosen meta datasets: \n{tgt_names}')
        # 3.load meta tasks and augment data
        database, topk = self.get_meta_database(all_proteins)
        meta_train = [database[name] for name in tgt_names]
        new = deepcopy(protein_tgt)
        new['df']['DMS_score'] = protein_tgt['df']['GEMME']
        new["df"] = new["df"].dropna(subset=["DMS_score"])
        new, _ = split_data(new, len(new['df']), shuffle=True)
        meta_train[-1:] = [new]

        if args.meta_tasks < 4:
            meta_train *= 2
        #meta_train为一个列表列表里面每个元素为一个字典即被选出的topk个数据集
        #train和test则是依据split_data在这个目标蛋白数据集上划分出来的数据集
        #返回一个字典，键为lr，train_loss，评估指标，最佳表现的周期,baseline
        print('Meta learning on anxilar datasets...')
        save_dir = self.get_save_dir('meta', train['name'])
        if os.path.exists(save_dir + '/logs.pkl'):
            print(f'{save_dir}/logs.pkl already exist, skipping!')
        else:
            _ = self.meta_single(meta_train, train, test, args.use_contrast)

        # meta transfer learning
        print('Meta transfer learning on target dataset...')
        save_dir = self.get_save_dir('meta-transfer', train['name'])
        if not use_bo:
            if os.path.exists(save_dir + '/logs.pkl'):
                print(f'{save_dir}/logs.pkl already exist, skipping!')
            else:
                _ = self.finetune_single_cv(train, test, use_contrast=args.use_contrast)
        else:
            _ = self.finetune_single_cv(train, test, use_contrast=args.use_contrast)

        # evaluation
        print('Evaluation on target dataset...')
        predicts_train, predcits_test, logits = self.test_single(train=train, test=test, 
                                    use_msa=args.use_msa, seq_aln_file=None)
        print('Meta transfer learning finished!')
        
        return predicts_train, predcits_test, logits.squeeze(0)
    
    def vectorize(self, protein_tgt, model='esm2'):
        model_name = config.model_dir[self.args.model]
        model = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        if model=='esm2':
            data = ProteinSequenceData([protein_tgt['wild_type']], tokenizer, device=self.device)
        else:
            struc_seqs = pd.read_csv(config.struc_seq_path, index_col='protein')
            sequence = struc_seqs.loc[protein_tgt['name'], 'struc_sequence']
            data = ProteinSequenceData([sequence], tokenizer, device=self.device)
        data_iter = DataLoader(data, batch_size=1, collate_fn=data.collate)
        prot2vec = Protein2Vector(model.to(self.device), pooling='average', hidden_fn=None)
        vector_tgt = prot2vec(data_iter)
        return vector_tgt
    
    def compute_nns(self, query_vecs, corpus_vecs, k):
        size = query_vecs.shape[0], corpus_vecs.shape[0], corpus_vecs.shape[1]
        query_vecs = query_vecs.unsqueeze(1).expand(*size)
        corpus_vecs = corpus_vecs.unsqueeze(0).expand(*size)
        data = TensorDataset(query_vecs, corpus_vecs)
        data_iter = DataLoader(data, batch_size=5)
        
        scores, indices = [], []
        for query_batch, corpus_batch in tqdm(data_iter, desc='Computing similarities...'):
            query_batch, corpus_batch = query_batch.to(self.device), corpus_batch.to(self.device)
            batch_scores = torch.cosine_similarity(query_batch, corpus_batch, -1)
            topk, topk_idx = batch_scores.topk(k, 1, largest=True)
            scores.extend(topk.tolist())
            indices.extend(topk_idx.tolist())
        return scores, indices
    

def directed_evolution_exp_mtl(
    protein: str,
    merged_data_path: str,
    round_base_path : str,
    round_file_names : List[str],
    gemme_data_path : str,
    wt_fasta_path: str,
    round_name: str,
    current_round: int,
    postfix: str = '_XXXX',
    total_rounds: int = 5,
    num_mutants_this_round: int = 20, 
    model_id: str = 'esm2',
    device: str = 'cuda:0',
    learning_strategy: str = 'topn', 
    output_dir : str = '',
):
    # 0.load data
    all_proteins = torch.load(merged_data_path)
    #print(all_proteins)
    
    # 1.load exp data
    all_experimental_data = []
    for round_file_name in round_file_names:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    
    # 2.load gemme data
    label_gemme, wt_seq = load_gemme_data(fasta_path=wt_fasta_path, score_path=gemme_data_path)

    # 3.initialization of training data and MTL Pipeline
    iteration, label = create_iteration_dataframes(all_experimental_data, label_gemme.variant.tolist())
    merged_df = pd.merge(
        left=label,
        right=label_gemme,
        how='left',
        left_on='mutant',
        right_on='variant'
    ).drop(columns='variant')
    merged_df.set_index('mutant', inplace=True)
    new_df, n_sites = defaultdict(list), set()
    for mutant, row in merged_df.iterrows():
        wt_aas, mt_aas, positions = '', '', []
        #有些DMS数据集为多突，形式为Q228R:T279L
        for site in mutant.split(':'): # handle multi-site mutants
            wt_aa, position, mt_aa = site[0], int(site[1:-1]) - 1, site[-1]
            assert wt_seq[position] == wt_aa, f'mismatch at {position}: {wt_seq[position]}!= {wt_aa}'
            #记录突变的原始氨基酸，突变氨基酸以及位置
            wt_aas += wt_aa
            mt_aas += mt_aa
            positions.append(position)
        
        new_df['wt_aas'].append(wt_aas)
        new_df['mt_aas'].append(mt_aas)
        new_df['positions'].append(tuple(positions))
        #记录此条记录突变个数
        n_sites.add(len(positions))
    #选取原df中的感兴趣的信息添加到新df
    new_df = pd.concat([pd.DataFrame(new_df, index=merged_df.index),
                        merged_df[['DMS_score', 'DMS_score_bin', 'GEMME']]], axis=1)
    #创建一个字典，仅含两个键一个是wtseq，一个是包含wt_aas, mt_aas, positions, DMS_score, DMS_score_bin的df
    protein_tgt = dict(wild_type=wt_seq, df=new_df)
    protein_tgt['offset'] = 0
    protein_tgt['n_sites'] = sorted(n_sites)
    protein_tgt['name'] = protein + postfix
    print("iterations considered\n", iteration)
    if learning_strategy == 'random' or learning_strategy == 'topn':
        if model_id == 'saprot':
            random_seed = str(current_round) + str(total_rounds)
            random_seed = int(random_seed)
            mtlpipeline = MTLPipeline_exp(
                al_config=al_config,
                train_size=len(iteration),
                protein=protein,
                seed=random_seed, 
                device=device,
                data_constructor=SaProtMutantData,
                score_fn=saprot_zero_shot,
                model=model_id
            )
        elif model_id == 'esm2':
            random_seed = str(current_round) + str(total_rounds)
            random_seed = int(random_seed)
            mtlpipeline = MTLPipeline_exp(
                al_config=al_config,
                train_size=len(iteration),
                protein=protein,
                seed=random_seed,
                device=device,
                model=model_id
            )
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")
    
        predicts_train, predicts_test, _ = mtlpipeline(all_proteins, iteration, protein_tgt ,postfix_topk='_AL', postfix=postfix)
        predicts_sorted_test = predicts_test.sort_values(by='prediction', ascending=False)
    else:
        ensemble_df = None
        for rep in range(1, 6):
            random_seed = str(current_round) + str(total_rounds) + str(rep)
            random_seed = int(random_seed)
            print(f'======================Rep {rep}======================')
            if model_id == 'saprot':
                mtlpipeline = MTLPipeline_exp(
                    al_config=al_config,
                    train_size=len(iteration),
                    protein=protein,
                    seed=random_seed, 
                    device=device,
                    data_constructor=SaProtMutantData,
                    score_fn=saprot_zero_shot,
                    model=model_id
                )
            elif model_id == 'esm2':
                mtlpipeline = MTLPipeline_exp(
                    al_config=al_config,
                    train_size=len(iteration),
                    protein=protein,
                    seed=random_seed,
                    device=device,
                    model=model_id
                )
            else:
                raise ValueError(f"Unsupported model_id: {model_id}")
            predicts_train, predicts_test, _ = mtlpipeline(all_proteins, iteration, protein_tgt ,postfix_topk='_AL', postfix=postfix, use_bo=True)
            predicts_sorted_test = predicts_test.sort_values(by='prediction', ascending=False)
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
    
    if learning_strategy == 'random':
        random_seed = str(current_round) + str(total_rounds)
        random_seed = int(random_seed)
        iteration_new_ids = random.sample(list(predicts_test.index), num_mutants_this_round)
    elif learning_strategy == 'topn':
        iteration_new_ids = predicts_test.sort_values(by='prediction', ascending=False).head(num_mutants_this_round).index.tolist()
    elif learning_strategy == 'ucb':
        adaptive_beta = 1-(current_round / total_rounds)
        result_df['UCB'] = result_df['prediction_mean'] + 5 * adaptive_beta * result_df['prediction_std']
        iteration_new_ids = result_df.sort_values(by='UCB', ascending=False).head(num_mutants_this_round).index.tolist()
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
        iteration_new_ids = result_df.sort_values(by='EI', ascending=False).head(num_mutants_this_round).index.tolist()
    
    # 5.Print results
    print(f"\nTested variants in this round: {len(iteration)}")
    print(iteration.variant.tolist())
    print(f"\nSelected variants in this round: {num_mutants_this_round}")
    print(predicts_sorted_test.loc[iteration_new_ids, :])
    
    # 6.Save results if an output_dir is provided
    if output_dir is not None:
        output_dir = os.path.join(output_dir, f'{protein}{postfix}', round_name)
        os.makedirs(output_dir, exist_ok=True)
        iteration.to_csv(os.path.join(output_dir, 'iteration.csv'))
        predicts_sorted_test.to_csv(os.path.join(output_dir, f'this_round_predictions_{learning_strategy}.csv'))
        predicts_sorted_test.loc[iteration_new_ids, :].to_csv(os.path.join(output_dir, f'selected_variants_{learning_strategy}.csv'))
        if learning_strategy == 'ucb':
            result_df.to_csv(os.path.join(output_dir, 'ucb_result.csv'))
        print(f"\nData saved to {output_dir}")
    
if __name__ == '__main__':
    
    directed_evolution_exp_mtl(
        protein='RsCas12f',
        merged_data_path='data/merged_AL.pkl',
        round_base_path='exp/round',
        round_file_names=['RsCas12f_Round1.xlsx', 'RsCas12f_Round2.xlsx', 'RsCas12f_Round3.xlsx'],
        gemme_data_path='exp/GEMME/RsCas12f/normPred_evolCombi.txt',
        wt_fasta_path='exp/seq/RsCas12f.fa',
        round_name='Round4',
        postfix='_XXXX',
        current_round=4,
        total_rounds=4,
        model_id='saprot',
        output_dir='exp/output',
        learning_strategy='topn'
    )
