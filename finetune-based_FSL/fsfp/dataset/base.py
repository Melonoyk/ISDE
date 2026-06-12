import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import sys

class ProteinSequenceData(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True, return_length=True)
        return sequences.to(self.device)

class MutantSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=False, device=None):
        if mask:
            self.sequences = {}
            for positions in set(protein['df']['positions']):
                mutant = list(protein['wild_type'])
                for position in positions: # get masked mutant sequence
                    mutant[position] = '<mask>'
                self.sequences[positions] = ''.join(mutant)
        else:
            self.sequences = [protein['wild_type']]
        
        for key, value in protein['df'].items():
            setattr(self, key, value.to_list())
        self.tokenizer = tokenizer
        self.device = device
        self.wt = [protein['wild_type']]
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx]
    
    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels = zip(*raw_batch)
        
        if type(self.sequences) is dict: # identify duplicate positions, possibly multi-site
            unique_pos = {pos: i for i, pos in enumerate(set(positions))}
            inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
            sequences = [self.sequences[pos] for pos in unique_pos.keys()]
        else:
            inv_idx = torch.zeros(len(positions), dtype=torch.long, device=self.device)
            sequences = self.sequences
        sequences = self.tokenizer(sequences, return_tensors='pt').to(self.device)
        wt_seq = self.tokenizer(self.wt, return_tensors='pt').to(self.device)
        
        positions = [torch.tensor(pos, device=self.device) + 1 for pos in positions]
        wt_aas = self.tokenizer(wt_aas, add_special_tokens=False)['input_ids']
        mt_aas = self.tokenizer(mt_aas, add_special_tokens=False)['input_ids']
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    inv_seq_idx=inv_idx,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas,
                    positions=positions,
                    targets=scores,
                    labels=labels,
                    wt_seq=wt_seq)

class MutantContrastiveSequenceData(Dataset):
    def __init__(self, protein, tokenizer, device=None):
        for key, value in protein['df'].items():
            setattr(self, key, value.to_list())
        self.sequences = {}
        for i, positions in enumerate(set(protein['df']['positions'])):
            mutant = list(protein['wild_type'])
            mt_aas = self.mt_aas[i]
            assert len(positions) == len(mt_aas), f'positions({len(positions)}) and mt_aas({len(mt_aas)}) length mismatch'
            for i, position in enumerate(positions): 
                mutant[position] = mt_aas[i]
            self.sequences[positions] = ''.join(mutant)
        self.tokenizer = tokenizer
        self.device = device
        self.wt = [protein['wild_type']]*len(self.positions)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.wt_aas[idx], self.mt_aas[idx], self.positions[idx], self.DMS_score[idx], self.DMS_score_bin[idx], self.wt[idx]
    
    def collate(self, raw_batch):
        wt_aas, mt_aas, positions, scores, labels, wt_sequences = zip(*raw_batch)
        
        unique_pos = {pos: i for i, pos in enumerate(set(positions))}
        inv_idx = torch.tensor([unique_pos[pos] for pos in positions], device=self.device)
        sequences = [self.sequences[pos] for pos in unique_pos.keys()]
        sequences = self.tokenizer(sequences, return_tensors='pt').to(self.device)
        wt_sequences = self.tokenizer(wt_sequences, return_tensors='pt').to(self.device)
        
        positions = [torch.tensor(pos, device=self.device) + 1 for pos in positions]
        wt_aas = self.tokenizer(wt_aas, add_special_tokens=False)['input_ids']
        mt_aas = self.tokenizer(mt_aas, add_special_tokens=False)['input_ids']
        scores = torch.tensor(scores, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return dict(sequences=sequences,
                    inv_seq_idx=inv_idx,
                    wt_aas=wt_aas,
                    mt_aas=mt_aas,
                    positions=positions,
                    targets=scores,
                    labels=labels,
                    wt_seq=wt_sequences)

class MetaContrastiveSequenceData(Dataset):
    def __init__(self, protein_splits, tokenizer, 
                 adapt_steps=5, mask='train', training=True,
                 constructor=MutantContrastiveSequenceData, device=None):
        self.support_iters = []
        self.query_iters = []
        for support, query in protein_splits:
            # random selection
            support = constructor(support, tokenizer,
                                          mask=mask in {'train', 'all'},
                                          device=device)
            # 计算每个支持集的批次数量
            total_support_data = len(support)
            adapt_batch_size = total_support_data // adapt_steps  # 每个批次包含的样本数
            support_iter = DataLoader(support,
                                      batch_size=adapt_batch_size,
                                      shuffle=True,
                                      collate_fn=support.collate)
            self.support_iters.append(support_iter)
            
            if training:
                query = constructor(query, tokenizer,
                                            mask=mask in {'train', 'all'},
                                            device=device)
            else:
                query = constructor(query, tokenizer, mask=mask in {'eval', 'all'}, device=device)
            eval_batch_size = len(query)
            query_iter = DataLoader(query,
                                     batch_size=eval_batch_size,
                                     collate_fn=query.collate)
            self.query_iters.append(query_iter)
        
    def __len__(self):
        return len(self.query_iters)
    
    def __getitem__(self, idx):
        adapt_batch = [batch for batch in self.support_iters[idx]]
        eval_batch = next(iter(self.query_iters[idx]))
        return adapt_batch, eval_batch
    
    def collate(self, raw_batch):
        adapt_batches, eval_batches = zip(*raw_batch)
        return dict(adapt_batches=adapt_batches,
                    eval_batches=eval_batches)

class RankingSequenceData(Dataset):
    def __init__(self, protein, tokenizer, mask=True, list_size=2, max_size=10000,
                 constructor=MutantSequenceData, device=None):
        self.mutant_data = constructor(protein, tokenizer, mask, device)
        self.list_size = list_size
        self.max_size = max_size
        self.device = device
        
        #计算可能的采样到的组合数
        total = math.comb(len(self.mutant_data), list_size)
        if max_size > total: # iteration over all combinations
            #如果可能采样到的组合数小于max_size,则直接把全部组合输出出来成为一个列表，列表长度为total内部的每个元素为长度为list_size的列表，记录着采样到的idx组合
            self.comb_idx = list(combinations(range(len(self.mutant_data)), list_size))
        #超出需求则随机采样
        else: # numerous combinations, random select instead
            self.comb_idx = None
    
    def __len__(self):
        if self.comb_idx is not None:
            return len(self.comb_idx)
        else:
            return self.max_size
    
    def __getitem__(self, idx): # yield combination indices instead of real data
        if self.comb_idx is not None:
            return self.comb_idx[idx]
        else:
            return random.sample(range(len(self.mutant_data)), self.list_size)
    
    def collate(self, comb_idx): # identify duplicate elements among a batch of combinations
        #转换为张量
        comb_idx = torch.tensor(comb_idx, device=self.device)
        unique_mt, inv_idx = torch.unique(comb_idx, return_inverse=True)

        #依据新顺序从mutant_data取出batch内的样本
        raw_batch = [self.mutant_data[i] for i in unique_mt]
        #使用MutantSequenceData的collate方法将raw_batch组织形成dict形式的batch结果
        batch = self.mutant_data.collate(raw_batch)
        batch['inv_list_idx'] = inv_idx
        return batch

class MetaRankingSequenceData(Dataset):
    def __init__(self, protein_splits, tokenizer, adapt_batch_size, eval_batch_size,
                 adapt_steps=5, mask='train', list_size=2, training=True,
                 constructor=MutantSequenceData, device=None):
        self.support_iters = []
        self.query_iters = []
        #遍历长度为topk的列表protein_splits，其中每个元素为一个元组，元组下为被split_data函数划分后的该数据集的训练集和测试集
        #又或者是经过K折交叉验证得到的长度为k的列表每个元素为一个元组，含有k折拆分的结果
        for support, query in protein_splits:
            # random selection
            #内循环的step由adapt_steps决定
            support = RankingSequenceData(support, tokenizer,
                                          mask=mask in {'train', 'all'},
                                          list_size=list_size,
                                          max_size=adapt_steps * adapt_batch_size,
                                          constructor=constructor,
                                          device=device)
            #调用RankingSequenceData生成一个用于排序的数据集并封装到Dataloader
            support_iter = DataLoader(support,
                                      batch_size=adapt_batch_size,
                                      shuffle=True,
                                      collate_fn=support.collate)
            self.support_iters.append(support_iter)
            #如果是训练期间，则需要一个ranking dataset进行listMLE的训练
            if training:
                query = RankingSequenceData(query, tokenizer,
                                            mask=mask in {'train', 'all'},
                                            list_size=list_size,
                                            max_size=eval_batch_size,
                                            constructor=constructor,
                                            device=device)
            #否则一个普通的数据集即可，用于简单的进行predict而无需计算ListMLEloss，详情见trainer.py的MetaRankingTrainer
            else:
                query = constructor(query, tokenizer, mask=mask in {'eval', 'all'}, device=device)
            #假如是training的话只能形成一个batch？
            query_iter = DataLoader(query,
                                     batch_size=eval_batch_size,
                                     collate_fn=query.collate)
            self.query_iters.append(query_iter)
        
    def __len__(self):
        return len(self.query_iters)
    
    def __getitem__(self, idx):
        #此处的idx是topk的个数，用于从列表support_iters和query_iters中取出指定数据集的支持集和查询集
        #并返回一个长度为该指定数据集长度的列表以便于取出其中指定批量大小的数据形成一个batch
        adapt_batch = [batch for batch in self.support_iters[idx]]
        eval_batch = next(iter(self.query_iters[idx]))
        return adapt_batch, eval_batch
    
    def collate(self, raw_batch):
        adapt_batches, eval_batches = zip(*raw_batch)
        return dict(adapt_batches=adapt_batches,
                    eval_batches=eval_batches)
