import numpy as np
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from kernels import CompositeKernel
from typing import Dict, Tuple, List
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from tqdm import trange
from evolvepro.src.data import DictToObject
import pandas as pd
import sys


class KermutGP(ExactGP):
    """Gaussian Process regression model for supervised variant effects predictions.

    A specialized Gaussian Process implementation that combines sequence and structural
    information for predicting the effects of protein mutations. It extends gpytorch's
    ExactGP class and supports both composite and single kernel architectures, as well
    as zero-shot prediction capabilities through its mean function.

    Args:
        train_inputs: Training input data for the GP model. Default expects tuple of
            (one-hot sequences, sequence_embeddings, zero-shot scores).
        train_targets: Target values corresponding to the training inputs.
        likelihood: Gaussian likelihood function for the GP model.
        kernel_cfg (DictConfig): Configuration dictionary for kernel specifications,
            containing settings for sequence_kernel and structure_kernel if composite
            is True, or a single kernel configuration if composite is False.
        use_zero_shot_mean (bool, optional): Whether to use a linear mean function
            for zero-shot predictions. If True, uses LinearMean; if False, uses
            ConstantMean. Defaults to True.
        composite (bool, optional): Whether to use a composite kernel combining
            sequence and structure information. If False, uses a single kernel
            specified in kernel_cfg. Defaults to True.
        **kwargs: Additional keyword arguments passed to the kernel initialization.

    Attributes:
        covar_module: The kernel (covariance) function, either a CompositeKernel
            or a single kernel as specified by kernel_cfg.
        mean_module: The mean function, either LinearMean for zero-shot predictions
            or ConstantMean for standard GP regression.
        use_zero_shot_mean (bool): Flag indicating whether zero-shot mean function
            is being used.
    """

    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        kernel_cfg: DictToObject,
        use_zero_shot_mean: bool = True,
        GP_inputs: Dict = None,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.covar_module = CompositeKernel(
            wt_sequence=GP_inputs['wt_sequence'],
            use_site_comparison=kernel_cfg.composite_kernel.use_site_comparison,
            use_mutation_comparison=kernel_cfg.composite_kernel.use_mutation_comparison,
            use_sequence_comparison=kernel_cfg.composite_kernel.use_sequence_comparison,
            use_matern_sequence_kernel=kernel_cfg.composite_kernel.use_matern_sequence_kernel,
            use_distance_comparison=kernel_cfg.composite_kernel.use_distance_comparison,
            use_rel_pos_comparison=kernel_cfg.composite_kernel.use_rel_pos_comparison,
            conditional_probs=GP_inputs.get('conditional_probs', None),
            coords=GP_inputs.get('coords', None),
            h_lengthscale=kernel_cfg.composite_kernel.h_lengthscale,
            d_lengthscale=kernel_cfg.composite_kernel.d_lengthscale,
            p_lengthscale=kernel_cfg.composite_kernel.p_lengthscale,
            nu=kernel_cfg.composite_kernel.nu,
        )

        self.use_zero_shot_mean = use_zero_shot_mean
        if self.use_zero_shot_mean:
            self.mean_module = LinearMean(input_size=1, bias=True)
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks, x_embed, x_zero=None) -> MultivariateNormal:
        #print(f'x_toks: {x_toks}\nx_embed: {x_embed}\nx_zero: {x_zero}')
        if x_zero is None:
            x_zero = x_toks
        x_zero = x_zero.float()
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        '''
        eigvals = torch.linalg.eigvalsh(covar_x)
        min_eigval = eigvals.min()
        print(f"Minimum Eigenvalue: {min_eigval.item():.4e}")  # 需 > 0

        # 条件数计算
        cond_number = torch.linalg.cond(covar_x)
        print(f"Condition Number: {cond_number.item():.4e}")
        '''
        min_eigval = torch.linalg.eigvalsh(covar_x).min()
        adaptive_jitter = max(1e-6, -min_eigval + 1e-6)  # 确保正定性
        covar_x += adaptive_jitter * torch.eye(covar_x.size(0), dtype=torch.float64).to(covar_x.device)

        return MultivariateNormal(mean_x, covar_x)

def instantiate_gp(
    cfg: DictToObject,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    gp_inputs: Dict,
    device: str = 'cuda:0'
) -> Tuple[KermutGP, GaussianLikelihood]:
    """Instantiates a KermutGP model and its associated Gaussian likelihood.

    Args:
        cfg: Configuration object containing model parameters including:
            - kernel.use_prior: Boolean indicating whether to use prior
            - kernel.noise_prior_scale: Scale parameter for HalfCauchy prior
            - kernel.use_structure_kernel: Boolean for structure kernel usage
            - kernel.use_sequence_kernel: Boolean for sequence kernel usage
            - kernel.use_zero_shot: Boolean for zero-shot mean
            - use_gpu: Boolean indicating GPU usage preference
        train_inputs: Tuple of torch tensors containing training input features.
            None values in the tuple will be filtered out.
        train_targets: Torch tensor containing training target values.
        gp_inputs: Dictionary containing additional inputs for the KermutGP model.

    Returns:
        A tuple containing:
            - KermutGP: The instantiated Gaussian Process model
            - GaussianLikelihood: The associated likelihood function

    Note:
        The function will automatically move the model and likelihood to GPU if
        cfg.use_gpu is True and a CUDA device is available.
    """

    if cfg.kernel.prior.use:
        noise_prior = HalfCauchyPrior(scale=cfg.kernel.prior.noise_prior_scale)
    else:
        noise_prior = None

    likelihood = GaussianLikelihood(noise_prior=noise_prior)
    train_inputs = tuple([x for x in train_inputs if x is not None])

    gp = KermutGP(
        train_inputs,
        train_targets,
        likelihood,
        kernel_cfg=cfg.kernel,
        use_zero_shot_mean=cfg.kernel.use_zero_shot,
        GP_inputs=gp_inputs
    )
    if cfg.other.use_gpu:
        gp = gp.to(device)
        likelihood = likelihood.to(device)

    return gp, likelihood

def optimize_gp(
    gp: ExactGP,
    likelihood: GaussianLikelihood,
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    lr: float = 3.0e-4,
    n_steps: int = 150,
    progress_bar: bool = True,
) -> Tuple[ExactGP, GaussianLikelihood]:
    """Optimizes a Gaussian Process using marginal likelihood maximization.

    Trains the GP model by minimizing the negative log marginal likelihood using
    the AdamW optimizer. The function handles training mode activation, optimizer
    configuration, and iterative optimization with optional progress bar display.

    Args:
        gp: The Gaussian Process model to be optimized. Must be an instance
            of ExactGP.
        likelihood: The Gaussian likelihood function associated with the GP model.
        train_inputs: Tuple of input tensors for training. None values in the tuple
            will be filtered out.
        train_targets: Target values tensor for training.
        lr: Learning rate for the AdamW optimizer. Default is 3.0e-4.
        n_steps: Number of optimization steps. Default is 150.
        progress_bar: Boolean controlling progress bar display. Default is True.

    Returns:
        A tuple containing:
            - ExactGP: The optimized Gaussian Process model
            - GaussianLikelihood: The optimized likelihood function

    Note:
        The function uses ExactMarginalLogLikelihood as the loss function and
        optimizes model parameters using gradient descent. The progress bar
        can be toggled through the configuration.
    """
    gp.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, gp)

    optimizer = torch.optim.AdamW(gp.parameters(), lr=lr)

    # None inputs not allowed
    x_train = tuple([x for x in train_inputs if x is not None])
    y_train = train_targets

    for _ in trange(n_steps, disable=not progress_bar):
        optimizer.zero_grad()
        output = gp(*x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    return gp, likelihood

def predict(
    gp: KermutGP,
    likelihood: GaussianLikelihood,
    inputs: tuple[torch.Tensor, ...],
) -> pd.DataFrame:
    """Makes predictions using a trained Gaussian Process and records results.

    Evaluates the GP model on test data and stores predictions, true values, and
    uncertainty estimates in a DataFrame. The function handles model evaluation mode,
    prediction generation, and proper CPU/numpy conversion of results.

    Args:
        gp: Trained KermutGP model to use for predictions.
        likelihood: Trained Gaussian likelihood function associated with the GP model.
        x_test: Tuple of input tensors for testing. None values in the tuple
            will be filtered out.
        y_test: Target values tensor for testing.
        test_fold: Integer indicating the current test fold number for cross-validation.
        test_idx: List of boolean values indicating which rows in df_out correspond
            to the current test set.
        df_out: DataFrame to store results, must contain columns:
            - 'fold': For cross-validation fold number
            - 'y': For true target values
            - 'y_pred': For predicted mean values
            - 'y_var': For prediction variances

    Returns:
        pd.DataFrame: Updated DataFrame containing the original data plus:
            - Model predictions (mean)
            - Prediction uncertainties (variance)
            - True values
            - Fold assignments

    """
    gp.eval()
    likelihood.eval()

    x_test = tuple([x for x in inputs if x is not None])

    with torch.no_grad():
        # Predictive distribution
        y_preds_dist = likelihood(gp(*x_test))
        y_preds_mean = y_preds_dist.mean.detach().cpu().numpy()
        y_preds_var = y_preds_dist.covariance_matrix.diag().detach().cpu().numpy()

    return y_preds_mean, y_preds_var

def top_layer(
    iter_train, 
    iter_test, 
    embeddings_pd, 
    labels_pd, 
    aux_pd,
    cfg,
    gp_inputs,
    measured_var, 
    final_round=10, 
    experimental=False
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
    y_train_activity_scaled = labels[iteration.isin(iter_train)]['activity_scaled']
    y_train_activity_binary = labels[iteration.isin(iter_train)]['activity_binary']

    if iter_test is not None:
        y_test = labels[iteration.isin([iter_test])][measured_var]
        print(y_test.shape)
        y_test_activity_scaled = labels[iteration.isin([iter_test])]['activity_scaled']
        y_test_activity_binary = labels[iteration.isin([iter_test])]['activity_binary']
    else:
        y_test = labels[iteration.isna()][measured_var]
        print(y_test.shape)
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
    if cfg.other.use_gpu and torch.cuda.is_available():
        x_toks_train, x_toks_test = x_toks_train.cuda(), x_toks_test.cuda()
        x_embed_train, x_embed_test = x_embed_train.cuda(), x_embed_test.cuda()
        x_zeroshot_train, x_zeroshot_test = x_zeroshot_train.cuda(), x_zeroshot_test.cuda()
        y_train, y_test = y_train.cuda(), y_test.cuda()

    train_inputs = (x_toks_train, x_embed_train, x_zeroshot_train)
    test_inputs = (x_toks_test, x_embed_test, x_zeroshot_test)
    train_targets = y_train
    test_targets = y_test
    
    gp, likelihood = instantiate_gp(
        cfg=cfg, 
        train_inputs=train_inputs,
        train_targets=train_targets, 
        gp_inputs=gp_inputs
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

    y_pred_test, y_pred_test_var = predict(
        gp=gp,
        likelihood=likelihood,
        inputs=test_inputs,
    )
    
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
        return this_round_variants, df_test, df_sorted_all
    else:
        return median_activity_scaled, top_activity_scaled, top_variant, top_final_round_variants, activity_binary_percentage, spearman_corr, ndcg, top_pr, df_test, this_round_variants


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
