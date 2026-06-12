import os
from typing import List, Dict, Any, Tuple, Union
from Bio import SeqIO
import pandas as pd
import numpy as np
import torch
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
residue_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

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
    position = int(variant[1:-1])
    return variant

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
        
        #Set iteration for WT in first round, exclude WT from subsequent rounds
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

    # Add missing variants to iter_test
    labels = add_missing_variants(labels, expected_variants)

    # Reorder iter_test based on expected variants
    labels = labels.set_index('variant').reindex(expected_variants, fill_value=np.nan).reset_index()
    labels.rename(columns={
        'variant': 'mutant',
        'activity':'DMS_score',
        'activity_binary':'DMS_score_bin',
        },inplace=True)
    
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
        'iteration': np.nan
    })
    return pd.concat([df, missing_df], ignore_index=True)