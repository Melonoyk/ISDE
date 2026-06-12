import numpy as np
import torch
from gaussion_process import instantiate_gp, optimize_gp, predict, KermutGP, GaussianLikelihood
import pandas as pd
import math
import sys

def top_layer_baseline(iter_train, iter_test, embeddings_pd, labels_pd, measured_var, model, experimental=False):
    
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
    model.fit(X_train, y_train)

    # make predictions on train data
    y_pred_train = model.predict(X_train)
    y_std_train = np.zeros(len(y_pred_train))
    # make predictions on test data
    # NOTE: can work on alternate 2-n round strategies here
    y_pred_test = model.predict(X_test)
    y_std_test = np.zeros(len(y_pred_test))

    return y_pred_test

def top_layer(
    iter_train, 
    iter_test, 
    embeddings_pd, 
    labels_pd, 
    aux_pd,
    cfg,
    gp_inputs,
    measured_var, 
    device='cuda:0',
    experimental=False,
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

    if iter_test is not None:
        y_test = labels[iteration.isin([iter_test])][measured_var]
        print(y_test.shape)
    else:
        y_test = labels[iteration.isna()][measured_var]
        print(y_test.shape)
    
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
    
    return y_pred_test


