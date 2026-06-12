import numpy as np
import torch
import pandas as pd
import math
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from gaussion_process import instantiate_gp, optimize_gp, predict, KermutGP, GaussianLikelihood
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm import tqdm
from utils import calc_ndcg, calc_toprecall
import sys
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
residue_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

def load_model(model_path):
    model = EsmForMaskedLM.from_pretrained(model_path)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    return model, tokenizer

def make_mutant_library(seq):
    """make mutant library for each site
    Args:
    seq: Target protein sequence eg.'MYKAG....'
    Returns:
    mutant_library: [(original_aa, site1, mutant_aa), (original_aa, site2, mutant_aa), ...]
    eg.[('M',2,'Y'),...]
    """
    mutant_library = []
    for i, aa in enumerate(seq):
        for sub_aa in residue_vocab:
            if sub_aa != aa:
                mutant = (aa, int(i+1), sub_aa)
                mutant_library.append(mutant)

    return mutant_library

def zero_shot_predicting(model, tokenizer, seq, device, labels_true=None, is_esm=True):
    mutant_library = make_mutant_library(seq)
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

    if labels_true is None:  
        sequence_library = sorted(sequence_library, key=lambda x: x[1], reverse=True)
        
        #calculate entropy of each position
        prob = torch.softmax(logits, dim=-1)
        entropies = {}
        for site in range(len(seq)):
            aa_probs = prob[site]
            entropy = -torch.sum(aa_probs * torch.log(aa_probs + 1e-10)).item()
            entropies[site] = entropy
        sorted_sites = sorted(entropies, key=entropies.get, reverse=True)
        selected_mutations_entropy = []
        for site in sorted_sites:
            site_mutations = [mutant for mutant in sequence_library if int(mutant[0][1:-1]) == site+1]
            if site_mutations:
                best_mutant = max(site_mutations, key=lambda x: x[1])
                selected_mutations_entropy.append(best_mutant)
        labels_entropy = pd.DataFrame(selected_mutations_entropy, columns=['mutant', 'prediction'])
    else:
        labels.set_index('mutant', inplace=True)
        labels_combined = labels_true.merge(
            labels, 
            left_on='variant',   # 左表连接列
            right_on='mutant',   # 右表连接列
            how='left'           # 左连接模式
        )

        # Calculate entropy-based labels (modified logic)
        prob = torch.softmax(logits, dim=-1)
        entropies = {site: -torch.sum(prob[site] * torch.log(prob[site] + 1e-10)).item() 
                    for site in range(len(seq))}
        # Generate entropy-based selection using aligned labels
        selected_mutations_entropy = []
        sorted_sites = sorted(entropies, key=entropies.get, reverse=True)
        for site in sorted(entropies, key=entropies.get, reverse=True):
            # Filter mutations present in labels_true (if provided)
            #site_mutations = labels_combined[labels_combined['variant'].str[1:-1].astype(int, errors='ignore') == (site + 1)]
            site_mutations = labels_combined[labels_combined['variant'].str[1:-1] == str(site + 1)]
            if site_mutations.empty:
                continue
            best_mutant = site_mutations.loc[site_mutations['prediction'].idxmax()]
            selected_mutations_entropy.append((best_mutant['variant'], best_mutant['prediction']))
        labels_entropy = pd.DataFrame(selected_mutations_entropy, columns=['variant', 'prediction'])

    return labels_combined, labels_entropy

def first_round(labels, labels_entropy=None, num_mutants_per_round=16, first_round_strategy='random',random_seed=None):
    if random_seed is not None:
            np.random.seed(random_seed)
    print("Starting labels length:", len(labels))
    variants = labels.variant
    # Perform random first round search strategy
    if first_round_strategy == 'random':
        random_mutants = np.random.choice(variants, size=num_mutants_per_round, replace=False)
        iteration_zero_ids = random_mutants

    elif first_round_strategy == 'topk':
        top_mutants_list = labels.sort_values(by='prediction', ascending=False).head(num_mutants_per_round)['variant'].tolist()
        iteration_zero_ids = top_mutants_list
        
    elif first_round_strategy == 'uncertainty':
        top_uncertain_mutant_list = labels_entropy.head(num_mutants_per_round)['variant'].tolist()
        iteration_zero_ids = top_uncertain_mutant_list
        
    else:
        print("Invalid first round search strategy.")
        return None, None

    # Create DataFrame for the first round
    iteration_zero = pd.DataFrame({'variant': iteration_zero_ids, 'iteration': 0})
    #WT = pd.DataFrame({'variant': 'WT', 'iteration': 0}, index=[0])
    #iteration_zero = pd.concat([iteration_zero, WT], ignore_index=True)
    this_round_variants = iteration_zero.variant
    labels_zero = pd.merge(labels, iteration_zero, on='variant', how='left')
    return labels_zero, iteration_zero, this_round_variants

def Thompson_sampling(
    gp: KermutGP,
    likelihood: GaussianLikelihood,
    inputs: tuple[torch.Tensor, ...],
    K: int, 
    iteration: int,
    total_iterations: int,
    initial_draw: int = None
):
    gp.eval(); likelihood.eval()
    x_test = tuple([x for x in inputs if x is not None])
    # 首次采样数 M = max(2K, K) 避免过少
    M = initial_draw or 4 * K
    print(f'M: {M}')
    selected_idxs = set()
    num_sample = int( 6 + 10 * (iteration / total_iterations))
    print(f'number of samples at iteration {iteration}: {num_sample}')

    with torch.no_grad():
        while len(selected_idxs) < K:
            # 1. 从后验联合分布中采样 M 条函数曲线
            #    参考：GPyTorch 支持多样本重参数化 .rsample(sample_shape) 
            posterior = likelihood(gp(*x_test))  # MultivariateNormal
            ts_samples = posterior.rsample(sample_shape=torch.Size([M]))  # [M, n]
            
            # 2. 转为 NumPy 并在每条曲线上找最优点索引
            #    np.argmax(axis=1) 返回每行（曲线）的最大值位置 :contentReference[oaicite:3]{index=3}
            ts_vals = ts_samples.detach().cpu().numpy()  # (M, n)
            #idxs = np.argmax(ts_vals, axis=1)         # (M,)
            idxs = np.argsort(-ts_vals, axis=1)[:, :num_sample]

            # 3. 将新索引加入集合，自动去重
            #selected_idxs.update(idxs.tolist())
            selected_idxs.update(idxs.ravel().tolist())
            print(f'selected_idx:\n{selected_idxs}')

            # 若仍不足 K，则增大采样量或继续循环
            if len(selected_idxs) < K:
                # 可以动态调整 M，比如加 50% 或保持不变
                num_sample = num_sample + 2
            #print(selected_idxs)
    # 4. 截取前 K 个不同索引，并映射到变体名称
    final_idxs = list(selected_idxs)[:K]
    return final_idxs

def top_layer(
    iter_train, 
    iter_test, 
    embeddings_pd, 
    labels_pd, 
    aux_pd,
    cfg,
    gp_inputs,
    measured_var, 
    current_iteration,
    total_iteration,
    final_round=10, 
    device='cuda:0',
    experimental=False,
    use_TS=False,
    is_ALDE=True
):  
    # if experimental, check alignment between embeddings and labels. This is done in the data loading for dms data
    if experimental:
        label_variants = labels_pd['variant'].tolist()
        embedding_variants = embeddings_pd.index.tolist()
        aux_variants = aux_pd.index.tolist()

        # Check if embedding row names and label variants are identical
        if label_variants == embedding_variants and label_variants == aux_variants:
            print('Embeddings, auxiliary data and labels are aligned')
        else:
            print('Embeddings, auxiliary data and labels are not aligned')
            print('Exiting.')
            return None
    
    #print(f'emb:\n{embeddings_pd}\naux:\n{aux_pd}\nlabels:\n{labels_pd}')
    
    # reset the indices of embeddings_pd and labels_pd
    embeddings_pd = embeddings_pd.reset_index(drop=True)
    labels_pd = labels_pd[labels_pd['variant'] != 'WT']
    labels_pd = labels_pd.reset_index(drop=True)
    aux_pd = aux_pd.reset_index(drop=True)    
    #print(f'emb:\n{embeddings_pd}\naux:\n{aux_pd}\nlabels:\n{labels_pd}')
    
    
    # save column 'iteration' in the labels dataframe
    iteration = labels_pd['iteration']
    #print(iteration)

    # save labels
    labels = labels_pd
    
    # save aux_pd
    aux = aux_pd

    # save mean embeddings as numpy array
    a = embeddings_pd

    # subset a, y to only include the rows where iteration = iter_train and iter_test
    idx_train = iteration[iteration.isin(iter_train)].index.to_numpy()
    if iter_test is not None:
        idx_test = iteration[iteration == iter_test].index.to_numpy()
    else:
        idx_test = iteration[iteration.isna()].index.to_numpy()

    #print(idx_test)
    # subset a to only include the rows where iteration = iter_train and iter_test
    emb_train = a.loc[idx_train, :]
    emb_test = a.loc[idx_test, :]
    aux_train = aux.loc[idx_train, :]
    aux_test = aux.loc[idx_test, :]
    x_toks_train, x_toks_test = torch.tensor(aux_train['x_toks'].tolist(), dtype=torch.long), torch.tensor(aux_test['x_toks'].tolist(), dtype=torch.long)
    x_embed_train, x_embed_test = torch.tensor(emb_train.values, dtype=torch.float64), torch.tensor(emb_test.values, dtype=torch.float64)
    #print(f'emb_train:\n{emb_train.shape}\nx_embed_trian:\n{x_embed_train.shape}')
    x_zeroshot_train, x_zeroshot_test = torch.tensor(aux_train['x_zeroshot'].values, dtype=torch.float64), torch.tensor(aux_test['x_zeroshot'].values, dtype=torch.float64)
    #print(f'x_toks_train:\n{x_toks_train}\nx_toks_test:\n{x_toks_test}\nx_embed_train:\n{x_embed_train}\nx_embed_test:\n{x_embed_test}\nx_zeroshot_train:\n{x_zeroshot_train}\nx_zeroshot_test:\n{x_zeroshot_test}')
    
    
    y_train = labels[iteration.isin(iter_train)][measured_var]
    if is_ALDE:
        y_train_activity_scaled = labels[iteration.isin(iter_train)]['activity_scaled']
        y_train_activity_binary = labels[iteration.isin(iter_train)]['activity_binary']

    if iter_test is not None:
        y_test = labels[iteration.isin([iter_test])][measured_var]
        print(y_test.shape)
        if is_ALDE:
            y_test_activity_scaled = labels[iteration.isin([iter_test])]['activity_scaled']
            y_test_activity_binary = labels[iteration.isin([iter_test])]['activity_binary']
    else:
        y_test = labels[iteration.isna()][measured_var]
        print(y_test.shape)
        if is_ALDE:
            y_test_activity_scaled = labels[iteration.isna()]['activity_scaled']
            y_test_activity_binary = labels[iteration.isna()]['activity_binary']       
    
    y_train_ori, y_test_ori = y_train.copy(), y_test.copy()
    y_train, y_test = torch.tensor(y_train.values, dtype=torch.float64), torch.tensor(y_test.values, dtype=torch.float64)

    # standardize
    mean = y_train.mean()
    std = y_train.std()
    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std
    del mean, std
    
    # move to gpu
    if cfg.other.use_gpu:
        x_toks_train = x_toks_train.to(device)
        x_embed_train = x_embed_train.to(device)
        x_zeroshot_train = x_zeroshot_train.to(device)
        y_train = y_train.to(device)

    
    train_inputs = (x_toks_train, x_embed_train, x_zeroshot_train)
    train_targets = y_train
    
    
    gp, likelihood = instantiate_gp(
        cfg=cfg, 
        train_inputs=train_inputs,
        train_targets=train_targets, 
        gp_inputs=gp_inputs,
        device=device
    )

    gp, likelihood = optimize_gp(
        gp=gp,
        likelihood=likelihood,
        train_inputs=train_inputs,
        train_targets=train_targets,
        lr=cfg.optim.lr,
        n_steps=cfg.optim.n_steps,
        progress_bar=cfg.optim.progress_bar,
    )

    y_pred_train, y_pred_train_var = predict(
        gp=gp,
        likelihood=likelihood,
        inputs=train_inputs,
    )
    
    if is_ALDE:
        x_toks_test= x_toks_test.to(device)
        x_embed_test = x_embed_test.to(device)
        x_zeroshot_test = x_zeroshot_test.to(device)
        y_test = y_test.to(device)
        test_inputs = (x_toks_test, x_embed_test, x_zeroshot_test)
        test_targets = y_test
        y_pred_test, y_pred_test_var = predict(
            gp=gp,
            likelihood=likelihood,
            inputs=test_inputs,
        )
    else:
        test_size = len(x_zeroshot_test)
        batch_size = 500
        n_batches = math.ceil(test_size / batch_size)
        batch_indices = [
            list(range(i*batch_size, min((i+1)*batch_size, test_size))) 
            for i in range(n_batches)
        ]
        
        y_pred_test = []
        for batch_idx in batch_indices:
            # 提取当前批次数据
            batch_toks = x_toks_test[batch_idx]
            batch_embed = x_embed_test[batch_idx]
            batch_zeroshot = x_zeroshot_test[batch_idx]
            batch_toks = batch_toks.to(device)
            batch_embed = batch_embed.to(device)
            batch_zeroshot = batch_zeroshot.to(device)
            batch_test_inputs = (batch_toks, batch_embed, batch_zeroshot)
        
            y_pred_test_batch, y_pred_test_var_batch = predict(
                gp=gp,
                likelihood=likelihood,
                inputs=batch_test_inputs,
            )
            y_pred_test.extend(y_pred_test_batch)
        
    # TThompson Sampling
    if is_ALDE:
        if use_TS:
            candidate_idx = Thompson_sampling(
                gp=gp,
                likelihood=likelihood,
                inputs=test_inputs,
                K=final_round,
                iteration=current_iteration,
                total_iterations=total_iteration,
            )
    
    if is_ALDE:
        df_train = pd.DataFrame({'variant': labels.variant[idx_train], 'y_pred': y_pred_train, 'y_actual': y_train_ori, 
                                'y_actual_scaled': y_train_activity_scaled, 'y_actual_binary': y_train_activity_binary,
                                'std_predictions': y_pred_train_var})
        df_test = pd.DataFrame({'variant': labels.variant[idx_test], 'y_pred': y_pred_test, 'y_actual': y_test_ori, 
                                'y_actual_scaled': y_test_activity_scaled, 'y_actual_binary': y_test_activity_binary,
                                'std_predictions': y_pred_test_var})
        df_all = pd.concat([df_train, df_test])
        
        df_sorted_all = df_all.sort_values('y_pred', ascending=False).reset_index(drop=True)
        df_sorted_test = df_test.sort_values('y_pred', ascending=False).reset_index(drop=True)
        df_sorted_train = df_train.sort_values('y_actual_scaled', ascending=False).reset_index(drop=True)
        # Get this round variants
        this_round_variants = df_train.variant


        # Calculate additional metrics
        median_activity_scaled = df_sorted_train.loc[:final_round, 'y_actual_scaled'].median()
        top_activity_scaled = df_sorted_train.loc[:final_round, 'y_actual_scaled'].max()
        top_variant = df_sorted_train.loc[df_sorted_train['y_actual_scaled'] == top_activity_scaled, 'variant'].values[0]
        top_final_round_variants = ",".join(df_sorted_train.loc[:final_round, 'variant'].tolist())
        spearman_corr = df_sorted_all[['y_pred', 'y_actual']].corr(method='spearman').iloc[0, 1]
        activity_binary_percentage = df_sorted_train.loc[:final_round, 'y_actual_binary'].mean()
        ndcg = calc_ndcg(np.array(df_sorted_all['y_actual']), np.array(df_sorted_all['y_pred']), quantile=False, top=10)
        top_pr = calc_toprecall(np.array(df_sorted_all['y_actual']), np.array(df_sorted_all['y_pred']), top_true=10, top_model=10)
        if experimental:
            if not use_TS:
                return this_round_variants, df_test, df_sorted_all
            else:
                return this_round_variants, df_test, df_sorted_all, candidate_idx
        else:
            if not use_TS:
                return median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, ndcg, top_pr, df_test, this_round_variants, None
            else:
                return median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, ndcg, top_pr, df_test, this_round_variants, candidate_idx
    else:
        if experimental:
            df_test = pd.DataFrame({'variant': labels.variant[idx_test], 'prediction': y_pred_test})
            return df_test.sort_values(by='prediction', ascending=False)
        else:
            return y_pred_test


def top_layer_baseline(iter_train, iter_test, embeddings_pd, labels_pd, measured_var, regression_type='randomforest', final_round=10, experimental=False):
    
    # if experimental, check alignment between embeddings and labels. This is done in the data loading for dms data
    if experimental:
        label_variants = labels_pd['variant'].tolist()
        embedding_variants = embeddings_pd.index.tolist()

        # Check if embedding row names and label variants are identical
        if label_variants == embedding_variants:
            print('Embeddings and labels are aligned')
        else:
            print('Embeddings and labels are not aligned')
            print('Exiting.')
            return None
    
    # reset the indices of embeddings_pd and labels_pd
    embeddings_pd = embeddings_pd.reset_index(drop=True)
    labels_pd = labels_pd.reset_index(drop=True)    

    # save column 'iteration' in the labels dataframe
    iteration = labels_pd['iteration']

    # save labels
    labels = labels_pd

    # save mean embeddings as numpy array
    a = embeddings_pd

    # subset a, y to only include the rows where iteration = iter_train and iter_test
    idx_train = iteration[iteration.isin(iter_train)].index.to_numpy()
    if iter_test is not None:
        idx_test = iteration[iteration == iter_test].index.to_numpy()
    else:
        idx_test = iteration[iteration.isna()].index.to_numpy()

    # subset a to only include the rows where iteration = iter_train and iter_test
    X_train = a.loc[idx_train, :]
    X_test = a.loc[idx_test, :]
    
    y_train = labels[iteration.isin(iter_train)][measured_var]

    if iter_test is not None:
        y_test = labels[iteration.isin([iter_test])][measured_var]
        print(y_test.shape)
    else:
        y_test = labels[iteration.isna()][measured_var]
        print(y_test.shape)   

    # fit
    if regression_type == 'ridge':
        model = linear_model.RidgeCV()
    elif regression_type == 'lasso':
        model = linear_model.LassoCV(max_iter=100000,tol=1e-3)
    elif regression_type == 'elasticnet':
        model = linear_model.ElasticNetCV(max_iter=100000,tol=1e-3)
    elif regression_type == 'linear':
        model = linear_model.LinearRegression(fit_intercept=False, positive=True)
    elif regression_type == 'neuralnet':
        model = MLPRegressor(hidden_layer_sizes=(5), max_iter=1000, activation='relu', solver='adam', alpha=0.001,
                             batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                             momentum=0.9, nesterovs_momentum=True, shuffle=True, random_state=1, tol=0.0001,
                             verbose=False, warm_start=False, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08)
    elif regression_type == 'randomforest':
        model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=5)
    elif regression_type == 'gradientboosting':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=1, min_samples_leaf=5)
    elif regression_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                    metric='minkowski', metric_params=None, n_jobs=None)
    elif regression_type == 'svm':
        model = SVR(kernel='rbf', C=100, gamma=1.0)

    model.fit(X_train, y_train)

    # make predictions on train data
    y_pred_train = model.predict(X_train)
    y_std_train = np.zeros(len(y_pred_train))
    # make predictions on test data
    # NOTE: can work on alternate 2-n round strategies here
    y_pred_test = model.predict(X_test)
    y_std_test = np.zeros(len(y_pred_test))

    return y_pred_test

