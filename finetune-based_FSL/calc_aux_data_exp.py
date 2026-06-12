import torch
import pandas as pd
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM
from itertools import chain
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
import numpy as np
import sys
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

residue_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

def load_model(model_path):
    model = EsmForMaskedLM.from_pretrained(model_path)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    return model, tokenizer

def make_mutant_library(seq):
    mutant_library = []
    for i, aa in enumerate(seq):
        for sub_aa in residue_vocab:
            if sub_aa != aa:
                mutant = (aa, int(i), sub_aa)
                mutant_library.append(mutant)
    return mutant_library

def zeroshot_predicting(
    model_path: str,
    seq: str,
    device: str,
    index: list
):
    model, tokenizer = load_model(model_path)
    device = 'cuda:0'
    model.to(device)

    wt_seq = tokenizer(seq, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = model(**wt_seq).logits.squeeze(0)
        log_prob = torch.log_softmax(logits, dim=-1)

    # get mutation probability
    aa_vocab = tokenizer.get_vocab()
    standard_aa = list("ACDEFGHIKLMNPQRSTVWY")  # 标准20种氨基酸（不含X）
    aa_indices = [aa_vocab[aa] for aa in standard_aa]
    
    prob_matrix = log_prob[1:-1, aa_indices].exp().cpu().numpy()

    # calculate mutant score
    sequence_library = []
    score_library = []
    all_records = []
    
    mutant_library = index
    
    for variant in mutant_library:
        mutated_sequence = list(seq)  # 转换为列表便于修改
        # 计算零样本预测分数
        predict = 0
        mutants = variant.split('_')
        for mutant in mutants:
            wt, pos, mut = mutant[0], int(mutant[1:-1]), mutant[-1]
            mutant_id = f"{wt}{str(pos)}{mut}"
            wt_aa = tokenizer(wt, add_special_tokens=False)['input_ids']
            mt_aa = tokenizer(mut, add_special_tokens=False)['input_ids']
            wt_prob = log_prob[pos-1, wt_aa]
            mt_prob = log_prob[pos-1, mt_aa]
            predict += (mt_prob - wt_prob)
            if mutated_sequence[pos-1] != wt:
                print(f"Warning: {mutant_id} position {pos} should be {wt}, but in wt sequence is {seq[pos-1]}")
                break
            # 执行氨基酸替换
            mutated_sequence[pos-1] = mut
        predict = float(predict)
        mutated_sequence = ''.join(mutated_sequence)
        record = SeqRecord(
                Seq(mutated_sequence),
                id=variant,
                description=""
            )
        all_records.append(record)
        score_library.append((variant, predict))
        sequence_library.append((variant, mutated_sequence))
                   
    labels_seq = pd.DataFrame(sequence_library, columns=['variant', 'seq'])
    labels_score = pd.DataFrame(score_library, columns=['variant', 'prediction'])
    return labels_seq, labels_score, prob_matrix, all_records

def extract_alpha_coords(
    protein_name: str,
    pdb_path: str,
    wt_seq: str,
) -> None:
    """提取并验证PDB与FASTA序列后保存alpha碳坐标"""
    pdb_path = Path(pdb_path)
    fasta_seq = wt_seq
    
    # 解析PDB文件
    parser = PDBParser()
    print(wt_seq)
    structure = parser.get_structure(protein_name, pdb_path)
    
    # 提取PDB序列
    pdb_seq = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":  # 仅处理氨基酸残基
                    res_name = residue.get_resname()
                    pdb_seq.append(seq1(res_name))
    pdb_seq = "".join(pdb_seq)
    print(pdb_seq)
    is_aligned = True
    # 序列一致性验证
    if pdb_seq != fasta_seq:
        print(
            f"Sequence mismatch :\nFASTA: {fasta_seq[:10]}...\nPDB: {pdb_seq[:10]}..."
        )
        is_aligned = False
        for i in range(len(fasta_seq)):
            if fasta_seq[i] != pdb_seq[i]:
                print(f"Mismatch at position {i+1}: FASTA={fasta_seq[i]}, PDB={pdb_seq[i]}")
                sys.exit()
    
    # 提取α碳坐标
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    try:
                        ca = residue["CA"]
                        ca_coords.append(ca.get_coord())
                    except KeyError:
                        continue  # 跳过缺失CA的残基

    if not is_aligned:
        ca_coords = ca_coords[:len(fasta_seq)]

    return np.array(ca_coords)

def extract_ProteinMPNN_probs(
    protein_name: str,
    raw_proteinmpnn_dir: str,
    wt_seq: str,
):
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}
    raw_proteinmpnn_dir = Path(raw_proteinmpnn_dir)
    wt_sequence = wt_seq

    file_path = (
        raw_proteinmpnn_dir
        / f"conditional_probs_only/{protein_name}.npz"
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
    

def generate_aux_data_exp(
    protein_name: str,
    fasta_path: str,
    pdb_path: str,
    raw_proteinmpnn_dir: str,
    emb_path: str,
    model_path: str = '/data1/users/weig03/data/pre_train_model/ESM/model/esm2_t33_650M_UR50D/',
    device: str = 'cuda:1',
    output_dir: str = 'output_proteingym/dms'
):

    output_csv_seq = f'{output_dir}/{protein_name}.csv'
    output_csv_score = f'{output_dir}/{protein_name}_zeroshot.csv'
    output_npy = f'{output_dir}/{protein_name}_condition_prob.npy'
    output_fasta = f'{output_dir}/{protein_name}.fasta'
    seq =  str(SeqIO.read(fasta_path, "fasta").seq)
    #print(protein['df'])
    
    if os.path.exists(output_csv_seq) and os.path.exists(output_npy) and os.path.exists(output_csv_score) and os.path.exists(output_fasta):
        print(f'{protein_name} already finished!')
    else:
        emb_df = pd.read_csv(emb_path, index_col=0)
        labels_seq, labels_score, prob_matrix, all_records = zeroshot_predicting(model_path, seq, device, emb_df.index.to_list())
        labels_seq.to_csv(output_csv_seq, index=False)
        labels_score.to_csv(output_csv_score, index=False)
        with open(output_fasta, "w") as handle:
            SeqIO.write(all_records, handle, "fasta")
        np.save(output_npy, prob_matrix)

    # extract alpha carbon coordinates
    print(f'Extracting alpha carbon coordinates')
    output_coord = f'{output_dir}/{protein_name}_ca_coords.npy'
    if os.path.exists(output_coord):
        print(f'{protein_name} already finished!')
    else:
        coord = extract_alpha_coords(protein_name, pdb_path, seq)
        np.save(output_coord, coord)

    #extract ProteinMPNN probs
    print(f'Extracting ProteinMPNN probs')
    output_proteinmpnn = f'{output_dir}/{protein_name}_proteinmpnn_probs.npy'
    if os.path.exists(output_proteinmpnn):
        print(f'{protein_name} already finished, skipping')
    else:
        proteinmpnn_probs = extract_ProteinMPNN_probs(protein_name, raw_proteinmpnn_dir, seq)
        np.save(output_proteinmpnn, proteinmpnn_probs)        
    

if __name__ == "__main__":
    generate_aux_data_exp(
        protein_name='RsCas12f',
        fasta_path='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/seq/RsCas12f.fa',
        pdb_path='/data1/users/weig03/data/Project/mutation_effect_pred/FSFP/exp/struc/RsCas12f.pdb',
        raw_proteinmpnn_dir='/data1/users/weig03/data/Focus_work/ProteinMPNN-main/output_exp',
        emb_path='exp/preprocess/RsCas12f/plm/RsCas12f_multimut_esm2_t33_650M_UR50D.csv',
        device='cuda:2',
        output_dir='exp/preprocess/RsCas12f'
    )
    
    a=np.load('exp/preprocess/RsCas12f/RsCas12f_condition_prob.npy')
    print(a.shape)
    b=np.load('exp/preprocess/RsCas12f/RsCas12f_proteinmpnn_probs.npy')
    print(b.shape)
    c=np.load('exp/preprocess/RsCas12f/RsCas12f_ca_coords.npy')
    print(c.shape)
    
