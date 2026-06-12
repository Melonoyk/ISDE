import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import scale, minmax_scale
from collections import defaultdict
from itertools import chain
from .data import make_dir, split_data
import random
from tqdm import tqdm
import sys

metrics = ['spearmanr', 'ndcg', 'topk_pr']
group_names = ['single_local', 'single_cross', 'single_rest',
               'multi_combined', 'multi_cross', 'multi_rest', 'all_rest']

def pairwise_ranking_loss(input1, input2, label1, label2, fn='hinge', margin=1.0):
    target = torch.where(label1 > label2, 1.0, -1.0)
    
    if fn == 'hinge':
        loss = F.margin_ranking_loss(input1, input2, target, margin=margin)
    elif fn == 'exp':
        loss = torch.exp(- target * (input1 - input2)).mean()
    elif fn == 'log':
        loss = torch.log(1 + torch.exp(- target * (input1 - input2))).mean()
    else:
        raise ValueError('Unknown pairwise ranking function: ' + fn)
    return loss

def listwise_ranking_loss(predicts, targets):
    ''' ListMLE loss '''
    indices = targets.sort(descending=True, dim=-1).indices
    predicts = torch.gather(predicts, dim=1, index=indices)

    cumsums = predicts.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    loss = torch.log(cumsums + 1e-10) - predicts
    return loss.sum(dim=1).mean()

def BT_loss(scores, golden_score):
    # scores 和 golden_score 现在的形状是 (batch_size, n)
    batch_size, n = scores.shape
    
    # 计算每个批次中的样本对之间的分数差异
    diff = scores.unsqueeze(1) - scores.unsqueeze(2)  # (batch_size, n, n)
    golden_diff = golden_score.unsqueeze(1) - golden_score.unsqueeze(2)  # (batch_size, n, n)
    
    # 计算损失：对于 golden_diff > 0，计算 exp(-diff)，对于 golden_diff <= 0，计算 exp(diff)
    loss_matrix = torch.log(1 + torch.exp(-diff)) * (golden_diff > 0).float() + \
                  torch.log(1 + torch.exp(diff)) * (golden_diff <= 0).float() 
    
    # 只计算上三角部分的损失，避免重复计算
    loss_matrix = torch.triu(loss_matrix, diagonal=1)  # 保留上三角，不包括对角线
    
    # 汇总所有损失
    loss = loss_matrix.sum(dim=[1, 2])  # 对于每个批次样本，求和
    
    # 归一化损失
    num_pairs = n * (n - 1) // 2  # 样本对的数量
    normalized_loss = loss / num_pairs  # 归一化损失
    
    # 返回每个批次的归一化损失
    return normalized_loss.mean()  # 对所有批次取平均

def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss

def compute_scores(predicts, targets, labels, k=30):
    report = dict(size=len(predicts))
    report['spearmanr'] = spearmanr(predicts, targets).statistic
    # std_tgts = scale([targets], axis=1)
    #std_tgts = minmax_scale([targets], (0, 5), axis=1)
    #report['ndcg'] = ndcg_score(std_tgts, [predicts],k=30)
    report['ndcg'] = calc_ndcg(np.array(targets), np.array(predicts), top=10)
    k = min(len(predicts), k)
    predicts, labels = torch.tensor(predicts), torch.tensor(labels)
    indices = predicts.topk(k).indices
    #report['topk_pr'] = torch.count_nonzero(labels[indices]).item() / k
    report['topk_pr'] = calc_toprecall(np.array(targets), np.array(predicts), top_true=10, top_model=10)
    return report

def group_scores(train_df, pred_df, test_df, k=30):
    #使用train_df的index如A3Y:A4E,转换为字符串并用split得到一个列表
    #每条record生成的突变位点列表再由于chain打开，并放入集合set
    #set存储了所有测试过的节点如A3Y，A4M，Y5E
    train_sites = set(chain(*train_df.index.str.split(':')))
    #取出positions记录的位置，为一个元组然后组成一个集合，记录了所有测试过的位置
    train_pos = set(chain(*train_df['positions']))
    
    groups = defaultdict(list)
    for mutant, row in test_df.iterrows():
        #判断该条record是单位点还是多位点
        n_sites = len(row['positions'])
        if n_sites == 1:
            #在single_rest中添加该record的行索引
            groups['single_rest'].append(mutant)
            if row['positions'][0] in train_pos:
                #如果这个单突变存在于train_pos则添加到single_local
                groups['single_local'].append(mutant)
            else:
                #否则放在single_cross
                groups['single_cross'].append(mutant)
        else:
            #添加到multi_rest
            groups['multi_rest'].append(mutant)
            sites = set(mutant.split(':'))
            if sites.issubset(train_sites):
                #如果是train_sites的子集则放在multi_combined
                groups['multi_combined'].append(mutant)
            elif sites.isdisjoint(train_sites):
                #如果两个集合不相交
                groups['multi_cross'].append(mutant)
    
    names, report = [], []
    for name in group_names:
        #取出该键对应的值
        indices = groups.get(name)
        if not indices or len(indices) < 3:
            continue
        #将非空且元素数大于等于3的键添加到names中
        names.append(name)
        #使用compute_scores将该键对应的records从pred_df和test_df中取出，计算spearman，ngdc，topk_pr并存为字典
        #存储到report列表中，于names列表一一对应
        report.append(compute_scores(pred_df.loc[indices].to_list(),
                                     test_df.loc[indices, 'DMS_score'].to_list(),
                                     test_df.loc[indices, 'DMS_score_bin'].to_list(),
                                     k))
    #计算一个整个数据集的得分，上面是各个类别的
    report.append(compute_scores(pred_df.loc[test_df.index].to_list(),
                                 test_df['DMS_score'].to_list(),
                                 test_df['DMS_score_bin'].to_list(),
                                 k))
    #刚刚最后算的对应name all_rest，生成一个行为不同组别，列为三个指标的df
    report = pd.DataFrame(report, index=names + ['all_rest'])
    return report, groups

def summarize_scores(score_groups, save_path=None):
    summary = {}
    #遍历每一个统计指标
    for metric in metrics:
        #遍历每一个score_groups.values即不同的蛋白质
        #遍历这个蛋白质对应的df的每一行，name为种类如single_local, multi_cross 等，row为三个统计值
        #取出对应的种类和想要的统计值组成一个字典，存放再reports列表中
        reports = [{name: row[metric] for name, row in groups.iterrows()} \
                       for groups in score_groups.values()]
        #转换为一个df，索引为种类，列为每个蛋白
        reports = pd.DataFrame(reports, index=score_groups.keys())
        reports.loc['average'] = reports.mean()
        summary[metric] = reports[[name for name in group_names if name in reports.columns]]
    
    if save_path is not None:
        make_dir(save_path)
        torch.save(summary, save_path)
    return summary

def read_multi_fasta(file_path):
    """
    params:
        file_path: path to a fasta file
    return:
        a dictionary of sequences
    """
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence.upper().replace('-', '<pad>').replace('.', '<pad>')
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences

def get_substitution_matrix(tokenizer, seq_aln_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seq_aln_file is not None:
        alignment_dict = read_multi_fasta(seq_aln_file)
    else:
        print('Not provide sequence alignment file, so does not use retrieval-enhanced strategy!')
        return None
    alignment_seqs = list(alignment_dict.values())
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split('/')[-1].split('-')
    except:
        aln_start, aln_end = 1, len(alignment_seqs[0])
    aln_start, aln_end = int(aln_start)-1, int(aln_end)
    print(f">>> Alignment start: {aln_start}, end: {aln_end}")
    print(f">>> Start tokenizing {len(alignment_seqs)} residue alignment sequences")
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_matrix = tokenized_results["input_ids"][:,1:-1]
    count_matrix = torch.zeros(alignment_matrix.size(1), tokenizer.vocab_size)
    for i in tqdm(range(alignment_matrix.size(1))):
        count_matrix[i] = torch.bincount(alignment_matrix[:,i], minlength=tokenizer.vocab_size)
    
    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
    count_matrix = torch.log_softmax(count_matrix, dim=-1)
    return [count_matrix, aln_start, aln_end]

def minmax(x):
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) ) 

def calc_ndcg(y_true, y_score, **kwargs):
    '''
    Inputs:
        y_true: an array of the true scores where higher score is better
        y_score: an array of the predicted scores where higher score is better
    Options:
        quantile: If True, uses the top k quantile of the distribution
        top: under the quantile setting this is the top quantile to
            keep in the gains calc. This is a PERCENTAGE (i.e input 10 for top 10%)
    Notes:
        Currently we're calculating NDCG on the continuous value of the DMS
        I tried it on the binary value as well and the metrics seemed mostly
        the same.
    '''
    if 'quantile' not in kwargs:
        kwargs['quantile'] = True
    if 'top' not in kwargs:
        kwargs['top'] = 10
    if kwargs['quantile']:
        k = np.floor(y_true.shape[0]*(kwargs['top']/100)).astype(int)
    else:
        k = kwargs['top']
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_score, pd.Series):
        y_score = y_score.values
    gains = minmax(y_true)
    ranks = np.argsort(np.argsort(-y_score)) + 1
    
    if k == 'all':
        k = len(ranks)

    #sub to top k
    ranks_k = ranks[ranks <= k]
    gains_k = gains[ranks <= k]
    #all terms with a gain of 0 go to 0
    ranks_fil = ranks_k[gains_k != 0]
    gains_fil = gains_k[gains_k != 0]
    
    #if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return (0)
    
    #discounted cumulative gains
    dcg = np.sum([g/np.log2(r+1) for r,g in zip(ranks_fil, gains_fil)])
    
    #ideal dcg - calculated based on the top k actual gains
    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    ideal_ranks_k = ideal_ranks[ideal_ranks <= k]
    ideal_gains_k = gains[ideal_ranks <= k]
    ideal_ranks_fil = ideal_ranks_k[ideal_gains_k != 0]
    ideal_gains_fil = ideal_gains_k[ideal_gains_k != 0]
    idcg = np.sum([g/np.log2(r+1) for r,g in zip(ideal_ranks_fil, ideal_gains_fil)])
    
    #normalize
    ndcg = dcg/idcg
    
    return (ndcg)

def calc_toprecall(true_scores, model_scores, top_true=10, top_model=10):  
    top_true = (true_scores >= np.percentile(true_scores, 100-top_true))
    top_model = (model_scores >= np.percentile(model_scores, 100-top_model))
    
    TP = (top_true) & (top_model)
    recall = TP.sum() / (top_true.sum()) if top_true.sum() > 0 else 0
    
    return (recall)
    
