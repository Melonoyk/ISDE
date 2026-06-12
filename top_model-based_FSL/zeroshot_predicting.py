import torch
import pandas as pd
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM

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

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
residue_vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']

model_path =f'/data1/users/weig03/data/pre_train_model/SaProt/SaProt_650M_AF2'
model, tokenizer = load_model(model_path)
device = 'cuda:0'
model.to(device)

seq = 'MdTkKdVwIdKkLwAfLwIdCaQ#Q#S#D#S#N#G#M#P#V#DdYpKvElVvNvKvIlLvWlEqLlQlRvQvTlRqEvIlKlNqKvSlIlQvYlClWvEvYlHvNvFvSqSvDvYvYcKvRvNpGvEdYgPdKdEcKcDvVvLqLvFdThLsGlGvYsVsNlDvKvFvKpTdGdNdDfLfYdSsAvNqCsSsTvTsVsRvGvVsCsGvEvFcKvNvScKvKvDcFcIvSvGvKvRdSpIrIdShYdKdEsNfQdPwLrDkLaHfNlKvSqIwRdLwEdYdSdDpHnEwFiYkViYwLgKqLtLgNdRpQvGnFcKvKvFsNnF#A#D#TnQtIgMmFiKtItLdVdRdDdNpSvTvKvTvIvLsEvRcCvLvDvEvVqYkSrVwSgAiSwKtLwIgYqDdKpKvKvKrCgWiVmLtNtLtSmYiShFhSdG#E#I#T#H#NpLfDdEaNlRwIeLkGfVwDdLaGdIqHqYwPnItCwAiSdVtYaGpEdWpKdRiFdTtIdDgGcGvEvIvEvEvYvRvRvRvVlEvAvRvKlKvTvLlLvKvQcGlKvNvCdGdDpGvRqIpGpHpGdVpKcTsRsNcKvPvVnYvSvIsEvDvRvInSvRvFvRlDlTvAvNlHlKvYvSlRvAvLvIlNvYvAcIvKvNvNrChGqVeIyQeMyEeNdL#E#G#V#T#A#H#S#D#K#F#L#K#N#W#S#YpYvDsLsQvTvKsInEvYvKvAsKvEsAnGnIhKhVyVhYyIdNhPlRpYqTlSlQqRaCaSsKvCpGrYdTgDdTpDvNqRpPvEdQpAqKwFgIaCdKpKpCpGgFdSiEdNgAsDsFnNsAsSnQnNcIrGnIdKpNpIsEvQvIvInKcEvEvIvQ#I#'
pure_seq = ''.join([seq[2*i] for i in range(len(seq)//2)])
mutant_library = make_mutant_library(pure_seq)
wt_seq = tokenizer(seq, return_tensors='pt').to(device)

with torch.no_grad():
    logits = model(**wt_seq).logits.squeeze(0)
    log_prob = torch.log_softmax(logits, dim=-1)

#calculate mutant score
sequence_library = []
for mutant in tqdm(mutant_library, desc='zero-shot predicting...'):
    mutant_score = 0
    wt, pos, mut = mutant
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
labels.to_csv(f'RsCas12f_0shot.csv', index=False)
    