import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Sequence
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
import sys
from transformers import EsmTokenizer  

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}

def load_dms_data(protein: dict, model_name: str, embeddings_path: str, 
                  embeddings_file_type: str = 'csv', embeddings_type_pt: str = 'both') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DMS data from files and align embeddings with labels.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model used for embeddings.
        embeddings_path (str): Path to the embeddings file.
        labels_path (str): Path to the labels file.
        embeddings_file_type (str): File type of embeddings ('csv' or 'pt').
        embeddings_type_pt (str, optional): Type of embeddings to use if 'pt' file ('average', 'mutated', or 'both'). Defaults to 'both'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned embeddings and labels DataFrames.
    """
    dataset_name = protein['name']
    # Generate file paths
    embeddings_file = os.path.join(embeddings_path, f'{dataset_name}_{model_name}.{embeddings_file_type}')
    #embeddings_file = os.path.join(embeddings_path, f'brenan_{model_name}.{embeddings_file_type}')

    # Load labels
    labels = protein['df']
    
    # Process embeddings based on file type
    if embeddings_file_type == "csv":
        embeddings = pd.read_csv(embeddings_file, index_col=0)
    elif embeddings_file_type == "pt":
        embeddings = torch.load(embeddings_file)
        embeddings = process_pt_embeddings(embeddings, embeddings_type_pt)
        if embeddings is None:
            return None, None
        embeddings = pd.DataFrame.from_dict(embeddings, orient='index')
    else:
        print("Invalid file type. Please choose either 'csv' or 'pt'")
        return None, None

    # Align embeddings with labels
    labels = labels.reset_index().rename(columns={'mutant': 'variant'})
    embeddings = embeddings[embeddings.index.isin(labels['variant'])]

    embeddings = embeddings.loc[labels['variant']]
    
    # Check if embeddings and labels are aligned
    if labels['variant'].tolist() == embeddings.index.tolist():
        print('Embeddings and labels are aligned')
        return embeddings, labels
    else:
        print('Embeddings and labels are not aligned')
        return None, None

def process_pt_embeddings(embeddings: Dict, embeddings_type_pt: str) -> Dict:
    """
    Process embeddings from .pt file based on the specified type.

    Args:
        embeddings (Dict): Dictionary containing embeddings.
        embeddings_type_pt (str): Type of embeddings to use ('average', 'mutated', or 'both').

    Returns:
        Dict: Processed embeddings dictionary.
    """
    # Get average or mutated embeddings
    if embeddings_type_pt == 'average':
        return {key: value['average'].numpy() for key, value in embeddings.items()}
    elif embeddings_type_pt == 'mutated':
        return {key: value['mutated'].numpy() for key, value in embeddings.items()}
    # Concatenate average and mutated embeddings
    elif embeddings_type_pt == 'both':
        return {key: np.concatenate((value['average'].numpy(), value['mutated'].numpy())) for key, value in embeddings.items()}
    else:
        print("Invalid embeddings_type_pt. Please choose 'average', 'mutated', or 'both'")
        return None

def load_experimental_embeddings(base_path: str, embeddings_file_name: str, rename_WT: bool = False) -> pd.DataFrame:
    """
    Load experimental embeddings from file.

    Args:
        base_path (str): Base path to the data directory.
        embeddings_file_name (str): Name of the embeddings file.

    Returns:
        pd.DataFrame: Experimental embeddings.
    """
    file_path = os.path.join(base_path, embeddings_file_name)
    embeddings = pd.read_csv(file_path, index_col=0)

    # Rename 'WT Wild-type sequence' to 'WT'
    if rename_WT:
        embeddings = embeddings.rename(index={'WT Wild-type sequence': 'WT'})

    return embeddings

def load_experimental_data(base_path: str, round_file_name: str, wt_fasta_path: str, single_mutant: bool = True) -> pd.DataFrame:
    """
    Load experimental data from file and process variants.

    Args:
        base_path (str): Base path to the data directory.
        round_file_name (str): Name of the round file.
        wt_fasta_path (str): Path to the wild-type FASTA file.
        single_mutant (bool, optional): Flag for single mutant processing. Defaults to True.

    Returns:
        pd.DataFrame: Processed experimental data.
    """
    # Load experimental data
    file_path = os.path.join(base_path, round_file_name)
    df = pd.read_excel(file_path)

    # Load wild-type sequence
    WT_sequence = str(SeqIO.read(wt_fasta_path, "fasta").seq)

    # Process variants
    if single_mutant:
        df['updated_variant'] = df['Variant'].apply(lambda x: process_variant(x, WT_sequence))
    else:
        df.rename(columns={'Variant': 'updated_variant'}, inplace=True)

    return df

def process_variant(variant: str, WT_sequence: str) -> str:
    """
    Process a single variant.

    Args:
        variant (str): Variant string.
        WT_sequence (str): Wild-type sequence.

    Returns:
        str: Processed variant string.
    """
    # Check if variant is WT
    if variant == 'WT':
        return variant
    
    # Extract position and amino acids
    position = int(variant[:-1])
    wt_aa = WT_sequence[position - 1]
    return wt_aa + variant

def create_iteration_dataframes(df_list: List[pd.DataFrame], expected_variants: List[str]) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    """
    Create training and testing dataframes for iterative learning.

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames containing experimental data from each round.
        expected_variants (List[str]): List of all expected variant names.

    Returns:
        Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]: 
            - iteration DataFrame: Contains variant and iteration information for training.
            - labels DataFrame: Contains variant, activity, and iteration information for testing.
            Returns (None, None) if duplicates are found.
    """
    processed_dfs = []

    # Process each round's data
    for round_num, df in enumerate(df_list, start=1):
        df_copy = df.copy()
        
        # Set iteration for WT in first round, exclude WT from subsequent rounds
        if round_num == 1:
            df_copy.loc[df_copy['updated_variant'] == 'WT', 'iteration'] = 0
        else:
            df_copy = df_copy[df_copy['updated_variant'] != 'WT']
        
        df_copy.loc[df_copy['updated_variant'] != 'WT', 'iteration'] = round_num
        df_copy['iteration'] = df_copy['iteration'].astype(float)
        df_copy.rename(columns={'updated_variant': 'variant'}, inplace=True)
        
        processed_dfs.append(df_copy)

    # Combine all processed dataframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Check for duplicates
    if has_duplicates(combined_df):
        return None, None

    # Create iter_train dataframe
    iteration = combined_df[['variant', 'iteration']]

    # Create iter_test dataframe
    labels = combined_df[['variant', 'activity', 'iteration']]

    # Add a activity_binary and activity_scaled column to labels
    labels['activity_binary'] = labels['activity'].apply(lambda x: 1 if x >= 1 else 0)
    labels['activity_scaled'] = labels['activity'].apply(lambda x: (x - labels['activity'].min()) / (labels['activity'].max() - labels['activity'].min()))

    # Add missing variants to iter_test
    labels = add_missing_variants(labels, expected_variants)

    # Reorder iter_test based on expected variants
    labels = labels.set_index('variant').reindex(expected_variants, fill_value=np.nan).reset_index()
    labels.rename(columns={'index': 'variant'}, inplace=True)
    
    return iteration, labels

def has_duplicates(df: pd.DataFrame) -> bool:
    """
    Check for duplicates in the 'variant' column of the dataframe.

    Args:
        df (pd.DataFrame): DataFrame to check for duplicates.

    Returns:
        bool: True if duplicates are found, False otherwise.
    """
    # Find duplicates in the 'variant' column
    duplicates = df[df.duplicated(subset=['variant'], keep=False)]
    
    # Print duplicates if found
    if not duplicates.empty:
        print("Duplicates found in variant column:")
        print(duplicates)
        print("Exiting.")
        return True
    return False

def add_missing_variants(df: pd.DataFrame, expected_variants: List[str]) -> pd.DataFrame:
    """
    Add missing variants to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to add missing variants to.
        expected_variants (List[str]): List of all expected variant names.

    Returns:
        pd.DataFrame: DataFrame with missing variants added.
    """
    missing_variants = set(expected_variants) - set(df['variant'])
    missing_df = pd.DataFrame({
        'variant': list(missing_variants),
        'activity': np.nan,
        'activity_binary': np.nan,
        'activity_scaled': np.nan,
        'iteration': np.nan
    })
    return pd.concat([df, missing_df], ignore_index=True)

def load_config(config_path: str) -> Dict:
    """Loads a YAML configuration file.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        Dict: Configuration settings as a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class DictToObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

def get_aux_data(
    labels_pd: pd.DataFrame,
    load_path: str,
    dataset_name: str)-> pd.DataFrame:
    """
    Load auxiliary data from a CSV file.
    Args:
        labels_pd (pd.DataFrame): DataFrame containing labels.
        load_path (str): Path to the CSV file.
        dataset_name (str): Name of the dataset.
    Returns:
        pd.DataFrame: DataFrame containing auxiliary data.
    """
    # Load auxiliary data from CSV file
    tokenizer = Tokenizer()
    
    #dataset_name = 'brenan'
    zeroshot_data = pd.read_csv(os.path.join(load_path, f'{dataset_name}_zeroshot.csv'))
    mut_seq_data = pd.read_csv(os.path.join(load_path, f'{dataset_name}.csv'))
    
    aux_df = pd.merge(
        left=zeroshot_data,
        right=mut_seq_data,
        on='variant',
        how='left'
    )
    labels_pd = labels_pd[labels_pd['variant'] != 'WT']
    aux_df.set_index('variant', inplace=True)
    aux_df = aux_df[aux_df.index.isin(labels_pd['variant'])]
    aux_df = aux_df.loc[labels_pd['variant']]
    
    x_toks = tokenizer(aux_df['seq'])
    aux_df['x_toks'] = x_toks.tolist()
    aux_df = aux_df.rename(columns={'prediction': 'x_zeroshot'})
    
    return aux_df

class Tokenizer:
    """Tokenizer for amino acid sequences. Converts sequences to one-hot encoded tensors."""

    def __init__(self, flatten: bool = True):
        super().__init__()
        # Uses the standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY
        self.alphabet = list(ALPHABET)
        self.flatten = flatten
        self._aa_to_tok = AA_TO_IDX
        self._tok_to_aa = {v: k for k, v in self._aa_to_tok.items()}

    def encode(self, batch: Sequence[str]) -> torch.LongTensor:
        batch_size = len(batch)
        seq_len = len(batch[0])
        toks = torch.zeros((batch_size, seq_len, 20))
        for i, seq in enumerate(batch):
            for j, aa in enumerate(seq):
                toks[i, j, self._aa_to_tok[aa]] = 1

        if self.flatten:
            # Check if batch is str
            if isinstance(batch, str):
                return toks.squeeze().flatten().long()
            else:
                return toks.reshape(batch_size, seq_len * 20).long()
        else:
            return toks.squeeze().long()

    def __call__(self, batch: Sequence[str]):
        return self.encode(batch)

def get_gp_inputs(
    dataset_name: str,
    load_path: str,
    cfg: DictToObject,
    seq: str,
    use_mpnn=False,
):
    inputs = {}
    tokenizer = Tokenizer()
    wt_sequence = seq
    wt_toks = tokenizer(wt_sequence)
    inputs["wt_sequence"] = wt_toks
    
    if (
        cfg.kernel.composite_kernel.use_site_comparison
        or cfg.kernel.composite_kernel.use_mutation_comparison
    ):
        if not use_mpnn:
            conditional_probs = np.load(Path(load_path) / f"{dataset_name}_condition_prob.npy")
            inputs["conditional_probs"] = torch.tensor(conditional_probs, dtype=torch.float32)
        else:
            conditional_probs = np.load(Path(load_path) / f"{dataset_name}_proteinmpnn_probs.npy")
            inputs["conditional_probs"] = torch.tensor(conditional_probs, dtype=torch.float32)
    if cfg.kernel.composite_kernel.use_distance_comparison:
        coords = np.load(Path(load_path) / f"{dataset_name}_ca_coords.npy")
        inputs["coords"] = torch.tensor(coords, dtype=torch.float64)
    #print(inputs["conditional_probs"])
    return inputs