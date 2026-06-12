import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from collections import defaultdict
from itertools import chain
from sklearn.preprocessing import StandardScaler

metrics = ['spearmanr', 'ndcg', 'topk_pr']
group_names = ['single_local', 'single_cross', 'single_rest',
               'multi_combined', 'multi_cross', 'multi_rest', 'all_rest']

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
    #print(k)
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

def summarize_scores(score_groups):
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
    
    return summary

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