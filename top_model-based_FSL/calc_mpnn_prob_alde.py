import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
import os

def extract_ProteinMPNN_probs(
    dataset_name: str
):
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}

    wt_fasta_path = f'./data/dms/wt_fasta/{dataset_name}_WT.fasta'
    seq = str(SeqIO.read(wt_fasta_path, "fasta").seq)
    wt_sequence = seq

    raw_proteinmpnn_dir = Path('./output/dms')
    file_path = (
        raw_proteinmpnn_dir
        / f"{dataset_name}.npz"
    )
    # Load and unpack
    raw_file = np.load(file_path)
    log_p = raw_file["log_p"]
    wt_toks = raw_file["S"]

    # Process logits ("X" is included as 21st AA in ProteinMPNN alphabet)
    #print(f'log_p.shape:\n{log_p.shape}')
    #print(log_p)
    log_p_mean = log_p.mean(axis=0)
    #print(f'log_p_mean.shape:\n{log_p_mean.shape}')
    #print(log_p_mean)
    p_mean = np.exp(log_p_mean)
    p_mean = p_mean[:, :20]
    #print(f'p_mean.shape:\n{p_mean.shape}')
    #print(p_mean)
    # Load sequence from ProteinMPNN outputs
    wt_seq_from_toks = "".join([proteinmpnn_tok_to_aa[tok] for tok in wt_toks])
    if wt_seq_from_toks!= wt_sequence:
        for i in range(len(wt_sequence)):
            if wt_seq_from_toks[i] != wt_sequence[i]:
                print(f"Mismatch at position {i+1}: FASTA={wt_sequence[i]}, MPNN={wt_seq_from_toks[i]}")
                sys.exit()
    
    return p_mean

if __name__ == '__main__':
    datasets = ["lee", "stiffler",  "cas12f", "brenan", "giacomelli", "zikv_E", "markin","haddox","doud", "kelsic", "cov2_S", "jones"]
    for dataset in datasets:
        output_proteinmpnn = f'./output/dms/{dataset}_proteinmpnn_probs.npy'
        if os.path.exists(output_proteinmpnn):
            print(f'Dataset: {dataset} already finished, skipping')
        else:
            p_mean = extract_ProteinMPNN_probs(dataset)
            np.save(output_proteinmpnn, p_mean)   