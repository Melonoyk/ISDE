import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from tqdm import tqdm

from Bio import SeqIO
from transformers import EsmTokenizer, EsmForMaskedLM

def make_mutant_library(seq):
    """make mutant library for each site
    Args:
    seq: Target protein sequence eg.'MYKAG....'
    Returns:
    mutant_library: [(original_aa, site1, mutant_aa), (original_aa, site2, mutant_aa), ...]
    eg.[('M',2,'Y'),...]
    """
    residue_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
    mutant_library = []
    for i, aa in enumerate(seq):
        for sub_aa in residue_vocab:
            if sub_aa != aa:
                mutant = (aa, int(i+1), sub_aa)
                mutant_library.append(mutant)

    return mutant_library

def extract_zeroshot_esm(
    model_path,
    fasta_path,
    output_dir,
    dataset_name,
):
    seq = str(SeqIO.read(fasta_path, "fasta").seq)
    mutant_library = make_mutant_library(seq)
    
    model = EsmForMaskedLM.from_pretrained(model_path)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    wt_seq = tokenizer(seq, return_tensors='pt').to(model.device)

    with torch.no_grad():
        logits = model(**wt_seq).logits.squeeze(0)
        log_prob = torch.log_softmax(logits, dim=-1)
    
    #calculate mutant score
    sequence_library = []
    for mutant in tqdm(mutant_library, desc='zero-shot predicting...'):
        mutant_score = 0
        wt, pos, mut = mutant
        assert seq[pos-1] == wt, f'error in making mutant library: {mutant} but {seq[pos-1]} != {wt}'
        wt_aa = tokenizer(wt, add_special_tokens=False)['input_ids']
        mt_aa = tokenizer(mut, add_special_tokens=False)['input_ids']
        mutant_score = log_prob[pos, mt_aa] - log_prob[pos, wt_aa]

        mutant_save = ''.join([wt, str(pos), mut])
        mutant_score = float(mutant_score)
        sequence_library.append((mutant_save, mutant_score))
    labels = pd.DataFrame(sequence_library, columns=['variant', 'prediction'])
    labels.to_csv(os.path.join(output_dir, f'{dataset_name}_zeroshot.csv'), index=False)
    
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
    labels_entropy = pd.DataFrame(selected_mutations_entropy, columns=['variant', 'prediction'])
    labels_entropy.to_csv(os.path.join(output_dir, f'{dataset_name}_zeroshot_entropy.csv'), index=False)
    

if __name__ == "__main__":
    #dataset_ls = ['cas12f', 'brenan', 'cov2_S', 'doud', 'giacomelli', 'haddox', 'jones', 'kelsic', 'lee', 'markin', 'stiffler', 'zikv_E']
    dataset_ls = ['markin']
    for dataset in dataset_ls:
        extract_zeroshot_esm(
            model_path='/data1/users/weig03/data/pre_train_model/ESM/model/esm2_t33_650M_UR50D',
            fasta_path=f'./data/dms/wt_fasta/{dataset}_WT.fasta',
            output_dir='./output/dms/',
            dataset_name=dataset
        )
    
