from Bio import SeqIO
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

def extract_condition_prob(
    model_path,
    fasta_path,
    output_dir,
    dataset_name,
):
    seq = str(SeqIO.read(fasta_path, "fasta").seq)
    
    model = EsmForMaskedLM.from_pretrained(model_path)
    for name, param in model.named_parameters():
        if 'contact_head.regression' in name:
            param.requires_grad = False
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    
    wt_seq = tokenizer(seq, return_tensors='pt').to(model.device)
    with torch.no_grad():
        logits = model(**wt_seq).logits.squeeze(0)
        log_prob = torch.log_softmax(logits, dim=-1)
    
    aa_vocab = tokenizer.get_vocab()
    standard_aa = list("ACDEFGHIKLMNPQRSTVWY")  # 标准20种氨基酸（不含X）
    aa_indices = [aa_vocab[aa] for aa in standard_aa]
    
    prob_matrix = log_prob[1:-1, aa_indices].exp().cpu().numpy()
    output_path = Path(output_dir) / f"{dataset_name}_condition_prob.npy"
    np.save(output_path, prob_matrix)

if __name__ == '__main__':
    dataset_ls = ['cas12f', 'brenan', 'cov2_S', 'doud', 'giacomelli', 'haddox', 'jones', 'kelsic', 'lee', 'markin', 'stiffler', 'zikv_E']
    for dataset in tqdm(dataset_ls, desc='extracting condition prob...'):
        extract_condition_prob(
            model_path='/data1/users/weig03/data/pre_train_model/ESM/model/esm2_t33_650M_UR50D',
            fasta_path=f'./data/dms/wt_fasta/{dataset}_WT.fasta',
            output_dir='./output/dms/',
            dataset_name=dataset,
        )
