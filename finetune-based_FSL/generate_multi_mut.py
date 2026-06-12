import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import combinations
from active_learning.utils.data import load_experimental_data
import csv
import sys

def generate_n_mutant_combinations(wt_fasta_path, round_base_path, round_file_names, n_series, output_file):
    """
    Generate a FASTA file containing combinations of n mutations, filtered by a threshold.

    Args:
        wt_fasta (str): Path to the FASTA file containing the wild-type protein sequence.
        mutant_file (str): Path to the Excel file containing mutant information.
        n (int): Number of mutations to combine.
        output_file (str): Path to the output FASTA file.
        threshold (float): Minimum value for including a mutant (default: 1).
    """
    # Read wild-type sequence
    wt_sequence = str(SeqIO.read(wt_fasta_path, "fasta").seq)
    
    # Read and process mutant data
    # load exp data
    all_experimental_data = []
    for round_file_name in round_file_names:
        experimental_data = load_experimental_data(round_base_path, round_file_name, wt_fasta_path)
        all_experimental_data.append(experimental_data)
        print(f"Loaded experimental data for {round_file_name}: {experimental_data.shape}")
    combined_df = pd.concat(all_experimental_data, ignore_index=True)
    mutants = combined_df[['Variant', 'activity']]
    
    # Create a SeqRecord object of the wild-type sequence
    mutant_combinations = []
    records = []
    for n in n_series:
        mutant_combinations += list(combinations(mutants['Variant'], n))

    # Generate mutant sequences for each combination
    for combination in mutant_combinations:
        positions = set()
        valid_combination = True
        mutant_sequence = wt_sequence
        variant = ""

        # Iterate over each mutant in the combination
        for mutant in combination:
            wt_aa, position, mutant_aa = mutant[0], mutant[1:-1], mutant[-1]
            i = int(position) - 1

            # Check if the position is already mutated
            if i in positions:
                valid_combination = False
                break

            # Update the mutant sequence and record the position
            positions.add(i)
            mutant_sequence = mutant_sequence[:i] + mutant_aa + mutant_sequence[i + 1:]
            variant += f'{wt_aa}{position}{mutant_aa}_'

        # Add the mutant sequence to the list of records if the combination is valid
        if valid_combination:
            record = SeqRecord(Seq(mutant_sequence), id=variant.rstrip('_'), description="")
            records.append(record)

    # Print the number of mutant combinations and valid mutant combinations
    print(f"Number of mutant combinations: {len(mutant_combinations)}")
    print(f"Number of valid mutant combinations: {len(records)}")

    # Write the mutant sequences to a FASTA file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as handle:
        SeqIO.write(records, handle, "fasta")
        
if __name__ == '__main__':
    wt_fasta_path = 'exp/seq/RsCas12f.fa'
    round_base_path = 'exp/round'
    round_file_names = ['RsCas12f_Round1.xlsx', 'RsCas12f_Round2.xlsx', 'RsCas12f_Round3.xlsx', 'RsCas12f_Round4.xlsx']
    n = [1,2]
    output_file = 'exp/preprocess/RsCas12f_Round5/RsCas12f.fasta'
    generate_n_mutant_combinations(wt_fasta_path, round_base_path, round_file_names, n, output_file)
