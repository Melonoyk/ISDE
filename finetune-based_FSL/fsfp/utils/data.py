import os
import pandas as pd
import torch
from collections import defaultdict
from itertools import chain
from sklearn.preprocessing import StandardScaler
import sys

def make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def trunc_sequence(protein, max_len):
    L = len(protein['wild_type'])
    if L <= max_len:
        protein['offset'] = 0
        return
    
    df = protein['df']
    positions = list(chain(*df['positions']))
    max_pos, min_pos = max(positions), min(positions)
    gap = max_pos - min_pos + 1
    
    if max_pos < max_len:
        protein['wild_type'] = protein['wild_type'][:max_len]
        protein['offset'] = 0
        return
    
    if gap <= max_len:
        window_l = max(min_pos - (max_len - gap) // 2, 0)
        window_r = min(max_pos + (max_len - gap) // 2, L - 1)
        seq_lr = protein['wild_type'][window_l: window_l + max_len]
        seq_rl = protein['wild_type'][window_r - max_len + 1: window_r + 1]
        
        if len(seq_lr) > len(seq_rl):
            protein['wild_type'] = seq_lr
            left, right = window_l, window_l + max_len
        else:
            protein['wild_type'] = seq_rl
            left, right = window_r - max_len + 1, window_r + 1
    else:
        n = 0
        left, right = min_pos, max_len
        window_l, window_r = min_pos, max_len
        while window_r < L:
            window_n = df['positions'].apply(
                lambda positions: all(window_l <= pos < window_r for pos in positions)).sum()
            if window_n > n:
                left, right = window_l, window_r
                n = window_n
            window_l += 1
            window_r += 1
        
        if right - left + 1 < max_len:
            left = right - max_len
        protein['wild_type'] = protein['wild_type'][left:right]
    
    df_bool = df.apply(lambda row: all(left <= pos < right for pos in row['positions']), axis=1)
    df = df.loc[df_bool].copy()
    df.loc[:, 'positions'] = df['positions'].apply(lambda positions: tuple(pos - left for pos in positions))
    protein['df'] = df
    protein['offset'] = left
    return

def process_dms(file_path, shuffle=True, max_len=1022, wild_type=None):
    df = pd.read_csv(file_path, index_col='mutant')
    if shuffle:
        #frac为抽取行的比例，1表示全部， sample抽样
        df = df.sample(frac=1)
   
    new_df, n_sites = defaultdict(list), set()
    for mutant, row in df.iterrows():
        wt_aas, mt_aas, positions = '', '', []
        #有些DMS数据集为多突，形式为Q228R:T279L
        for site in mutant.split(':'): # handle multi-site mutants
            wt_aa, position, mt_aa = site[0], int(site[1:-1]) - 1, site[-1]
            if wild_type is None:
                assert row['mutated_sequence'][position] == mt_aa, f'mismatch at {position}: {row["mutated_sequence"][position]} != {mt_aa}'
            else:
                assert wild_type[position] == wt_aa, f'mismatch at dataset{file_path} {position}: {wild_type[position]}!= {wt_aa}'
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
    new_df = pd.concat([pd.DataFrame(new_df, index=df.index),
                        df[['DMS_score', 'DMS_score_bin']]], axis=1)
    if wild_type is None:
        wild_type = list(row['mutated_sequence'])
        for wt_aa, position in zip(wt_aas, positions): # recover wild type sequence
            wild_type[position] = wt_aa
        #从突变序列结果恢复到wt序列
        wild_type = ''.join(wild_type)
    #创建一个字典，仅含两个键一个是wtseq，一个是包含wt_aas, mt_aas, positions, DMS_score, DMS_score_bin的df
    protein = dict(wild_type=wild_type, df=new_df)
    #根据指定的有效长度截取序列，同时向protein字典添加键offset以表示截取起始位置
    trunc_sequence(protein, max_len)
    protein['n_sites'] = sorted(n_sites)
    protein['name'] = os.path.basename(file_path).split('.')[0]
    return protein

def merge_files(data_dir, shuffle=True, max_len=1022, save_path=None):
    file_names = os.listdir(data_dir)
    #创造一个大字典，统一存储所有的数据集，键为蛋白名字，值为process_dms的输出（字典，键为wtseq，df，offset，n_sites，name）
    proteins = defaultdict(list)
    aux_fasta_data = 'data/aux_fatsa.csv'
    aux_wt_seqs = pd.read_csv(aux_fasta_data, index_col='protein')
    for file_name in file_names:
        
        if 'indels' in file_name:
            continue
        name = '_'.join(file_name.split('_')[:2])
        dataset = file_name.split('.')[0]
        if dataset in aux_wt_seqs.index:
            wildtype = aux_wt_seqs.loc[dataset, 'sequence']
        else:
            wildtype = None
        protein = process_dms(f'{data_dir}/{file_name}', shuffle, max_len, wildtype)
        if wildtype is not None:
            print(protein)
        
        proteins[name].append(protein)
    
    if save_path is not None:
        make_dir(save_path)
        torch.save(proteins, save_path)
    return proteins

def normalize(train_df, test_df):
    train_scores = train_df['DMS_score'].to_numpy()[:,None]
    test_scores = test_df['DMS_score'].to_numpy()[:,None]
    scaler = StandardScaler()
    train_df['DMS_score'] = scaler.fit_transform(train_scores).squeeze(1)
    test_df['DMS_score'] = scaler.transform(test_scores).squeeze(1)

def split_data(protein, train_size=0.8, shuffle=False, n_sites=None, neg_train=False,
               scale=False, train_ids=None):
    df = protein['df']
    train, test = protein.copy(), protein.copy()
    
    if train_ids is not None:
        train['df'] = df.loc[train_ids]
        test['df'] = df.loc[df.index.difference(train_ids, sort=False)]
    else:
        N = len(df)
        if train_size < 1:
            train_size = int(N * train_size)
        if shuffle:
            df = df.sample(frac=1)
        if n_sites is not None:
            n_sites = set(n_sites)
    
        df_bool = df.apply(lambda row: (not n_sites or len(row['positions']) in n_sites) and \
                                       (not neg_train or row['DMS_score_bin'] == 0), axis=1)
        train['df'] = df.loc[df_bool].iloc[:train_size]
        test['df'] = df.loc[df.index.difference(train['df'].index, sort=False)]
    
    if scale:
        normalize(train['df'], test['df'])
    return train, test
