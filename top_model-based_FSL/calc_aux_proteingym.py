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

def zeroshot_predicting(
    model_path: str,
    seq: str,
    protein_df: pd.DataFrame,
    device: str,
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
    for mutant_id, row in protein_df.iterrows():
        mutated_sequence = list(seq)  # 转换为列表便于修改
        # 计算零样本预测分数
        positions, wt_aas, mt_aas  = row['positions'], row['wt_aas'], row['mt_aas']
        predict = 0
        for pos, wt, mut in zip(positions, wt_aas, mt_aas):
            wt_aa = tokenizer(wt, add_special_tokens=False)['input_ids']
            mt_aa = tokenizer(mut, add_special_tokens=False)['input_ids']
            wt_prob = log_prob[pos+1, wt_aa]
            mt_prob = log_prob[pos+1, mt_aa]
            predict += (mt_prob - wt_prob)
            if mutated_sequence[pos] != wt:
                print(f"Warning: {mutant_id} position {pos+1} should be {wt_aa}, but in wt sequence is {seq[pos+1]}")
                break
            # 执行氨基酸替换
            mutated_sequence[pos] = mut
        predict = float(predict)
        mutated_sequence = ''.join(mutated_sequence)
        record = SeqRecord(
                Seq(mutated_sequence),
                id=mutant_id,
                description=""
            )
        all_records.append(record)
        score_library.append((mutant_id, predict))
        sequence_library.append((mutant_id, mutated_sequence))
                   
    labels_seq = pd.DataFrame(sequence_library, columns=['variant', 'seq'])
    labels_score = pd.DataFrame(score_library, columns=['variant', 'prediction'])
    return labels_seq, labels_score, prob_matrix, all_records

def extract_alpha_coords(
    dataset_name: str,
    wt_seq: str,
) -> None:
    """提取并验证PDB与FASTA序列后保存alpha碳坐标"""
    # 构造文件路径
    parts = dataset_name.strip().split('_')
    protein_name = parts[0] + '_' + parts[1]
    pdb_path = Path(f"data/proteingym/pdb_proteingym/{protein_name}.pdb")
    if dataset_name == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
        fasta_seq = 'IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYCYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKGSLVTCAKFTCSKKMTGKSIQPENLEYRIMLSVHGSQHSGMIVNDTGYETDENRAKVEVTPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLPWHAGADTGTPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKGKLFSGHLKCRLKMDKLRLKGVSYSLCTAAFTFTKVPAETLHGTVTVEVQYAGTDGPCKIPVQMAVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGDKKITHHWHRSGSTIGKAFEATVRGAKRMAVLGDTAWDFGSVGGVFNSLGKGIHQIFGAAFKSLFGGMSWFSQILIGTLLVWLGLNTKNGSISLTCLALGGVMIFLSTAVSA'
    elif dataset_name == 'POLG_HCVJF_Qi_2014':
        fasta_seq = 'RDVWDWVCTILTDFKNWLTSKLFPKLPGLPFISCQKGYKGVWAGTGIMTTRCPCGANISGNVRLGSMRITGPKTCMNTWQGTFPINCYTEGQCAPKPPTNYKTAIWRVAASEYAEVTQHGSYSYVTGLTTDNLKIPCQLPSPEFFSWVDGVQIHRFAPTPKPFFRDEVSFCVGLNSYAVGSQLPCEPEPDADVLRSMLTDPPHITAETAARRLARGSPPSEASSSVSQLSAPSLRATCTTHSNT'
    else:
        fasta_seq = wt_seq
    
    # 解析PDB文件
    parser = PDBParser()
    print(wt_seq)
    structure = parser.get_structure(dataset_name, pdb_path)
    
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
    if dataset_name == 'BRCA1_HUMAN_Findlay_2018':
        pdb_seq = pdb_seq[833:1855]
    elif dataset_name == 'UBE4B_MOUSE_Starita_2013':
        pdb_seq = pdb_seq[151:]
    elif dataset_name == 'SCN5A_HUMAN_Glazer_2019':
        pdb_seq = pdb_seq[994:]
    elif dataset_name.startswith('P53_HUMAN_Giacomelli_') :
        pdb_seq = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
    is_aligned = True
    # 序列一致性验证
    if pdb_seq != fasta_seq:
        print(
            f"Sequence mismatch at dataset:{dataset_name}!\nFASTA: {fasta_seq[:10]}...\nPDB: {pdb_seq[:10]}..."
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
    if dataset_name == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
        ahead_coord = [np.zeros(3, dtype=np.float32) for _ in range(290)]
        behind_coord = [np.zeros(3, dtype=np.float32) for _ in range(228)]
        ca_coords = ahead_coord + ca_coords + behind_coord
    elif dataset_name == 'BRCA1_HUMAN_Findlay_2018':
        assert len(ca_coords)==1863, f"BRCA1_HUMAN_Findlay_2018 length error: {len(ca_coords)}"
        ca_coords = ca_coords[833:1855]
    elif dataset_name == 'UBE4B_MOUSE_Starita_2013':
        assert len(ca_coords)==1173, f"UBE4B_MOUSE_Starita_2013 length error: {len(ca_coords)}"
        ca_coords = ca_coords[151:]
    elif dataset_name == 'POLG_HCVJF_Qi_2014':
        assert len(ca_coords)==244, f"POLG_HCVJF_Qi_2014 length error: {len(ca_coords)}"
        ahead_coord = [np.zeros(3, dtype=np.float32) for _ in range(456)]
        behind_coord = [np.zeros(3, dtype=np.float32) for _ in range(322)]
        ca_coords = ahead_coord + ca_coords + behind_coord
    elif dataset_name == 'SCN5A_HUMAN_Glazer_2019':
        assert len(ca_coords)==2016, f"SCN5A_HUMAN_Glazer_2019 length error: {len(ca_coords)}"
        ca_coords = ca_coords[994:]
    return np.array(ca_coords)

def extract_ProteinMPNN_probs(dataset_name: str,
    wt_seq: str,):
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}
    raw_proteinmpnn_dir = Path('data/proteingym/condition_prob_mpnn')

    parts = dataset_name.strip().split('_')
    protein_name = parts[0] + '_' + parts[1]
    UniProt_ID = protein_name
    DMS_id = dataset_name
    if dataset_name == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
        wt_sequence = 'IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYCYEASISDMASDSRCPTQGEAYLDKQSDTQYVCKRTLVDRGWGNGCGLFGKGSLVTCAKFTCSKKMTGKSIQPENLEYRIMLSVHGSQHSGMIVNDTGYETDENRAKVEVTPNSPRAEATLGGFGSLGLDCEPRTGLDFSDLYYLTMNNKHWLVHKEWFHDIPLPWHAGADTGTPHWNNKEALVEFKDAHAKRQTVVVLGSQEGAVHTALAGALEAEMDGAKGKLFSGHLKCRLKMDKLRLKGVSYSLCTAAFTFTKVPAETLHGTVTVEVQYAGTDGPCKIPVQMAVDMQTLTPVGRLITANPVITESTENSKMMLELDPPFGDSYIVIGVGDKKITHHWHRSGSTIGKAFEATVRGAKRMAVLGDTAWDFGSVGGVFNSLGKGIHQIFGAAFKSLFGGMSWFSQILIGTLLVWLGLNTKNGSISLTCLALGGVMIFLSTAVSA'
    elif dataset_name == 'POLG_HCVJF_Qi_2014':
        wt_sequence = 'RDVWDWVCTILTDFKNWLTSKLFPKLPGLPFISCQKGYKGVWAGTGIMTTRCPCGANISGNVRLGSMRITGPKTCMNTWQGTFPINCYTEGQCAPKPPTNYKTAIWRVAASEYAEVTQHGSYSYVTGLTTDNLKIPCQLPSPEFFSWVDGVQIHRFAPTPKPFFRDEVSFCVGLNSYAVGSQLPCEPEPDADVLRSMLTDPPHITAETAARRLARGSPPSEASSSVSQLSAPSLRATCTTHSNT'
    else:
        wt_sequence = wt_seq

    file_path = (
        raw_proteinmpnn_dir
        / UniProt_ID
        / f"conditional_probs_only/{UniProt_ID}.npz"
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

    # Mismatch between WT and PDB
    if DMS_id == "CAS9_STRP1_Spencer_2017_positive":
        p_mean = p_mean[:1368]
        wt_seq_from_toks = wt_seq_from_toks[:1368]
    if DMS_id in [
        "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
        "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
        "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
    ]:
        # Replace index 71 with "R"
        wt_seq_from_toks = wt_seq_from_toks[:71] + "R" + wt_seq_from_toks[72:]

    if dataset_name == 'BRCA1_HUMAN_Findlay_2018':
        wt_seq_from_toks = wt_seq_from_toks[833:1855]
    elif dataset_name == 'UBE4B_MOUSE_Starita_2013':
        wt_seq_from_toks = wt_seq_from_toks[151:]
    elif dataset_name == 'SCN5A_HUMAN_Glazer_2019':
        wt_seq_from_toks = wt_seq_from_toks[994:]
    elif dataset_name.startswith('P53_HUMAN_Giacomelli_') :
        wt_seq_from_toks = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
    # Special case where PDB is domain of a larger protein
    if dataset_name == 'A0A140D2T1_ZIKV_Sourisseau_growth_2019':
        ahead_cond = [np.zeros(20, dtype=np.float32) for _ in range(290)]
        behind_cond = [np.zeros(20, dtype=np.float32) for _ in range(228)]
        p_mean = np.concatenate([ahead_cond, p_mean, behind_cond], axis=0)
    elif dataset_name == 'BRCA1_HUMAN_Findlay_2018':
        assert len(p_mean)==1863, f"BRCA1_HUMAN_Findlay_2018 length error: {len(p_mean)}"
        p_mean = p_mean[833:1855]
    elif dataset_name == 'UBE4B_MOUSE_Starita_2013':
        assert len(p_mean)==1173, f"UBE4B_MOUSE_Starita_2013 length error: {len(p_mean)}"
        p_mean = p_mean[151:]
    elif dataset_name == 'POLG_HCVJF_Qi_2014':
        assert len(p_mean)==244, f"POLG_HCVJF_Qi_2014 length error: {len(p_mean)}"
        ahead_cond = [np.zeros(20, dtype=np.float32) for _ in range(456)]
        behind_cond = [np.zeros(20, dtype=np.float32) for _ in range(322)]
        p_mean = np.concatenate([ahead_cond, p_mean, behind_cond], axis=0)
    elif dataset_name == 'SCN5A_HUMAN_Glazer_2019':
        assert len(p_mean)==2016, f"SCN5A_HUMAN_Glazer_2019 length error: {len(p_mean)}"
        p_mean = p_mean[994:]
    else:
        if wt_seq_from_toks!= wt_sequence:
            for i in range(len(wt_sequence)):
                if wt_seq_from_toks[i] != wt_sequence[i]:
                    print(f"Mismatch at position {i+1}: FASTA={wt_sequence[i]}, MPNN={wt_seq_from_toks[i]}")
                    sys.exit()
    
    return p_mean
    

def generate_aux_data(
    protein_name: str,
    model_path: str = '/data1/users/weig03/data/pre_train_model/ESM/model/esm2_t33_650M_UR50D/',
    device: str = 'cuda:1',
    work_dir_base: str = 'output_proteingym/dms'
):
    path = f'data/proteingym/merged.pkl'
    datasets = torch.load(path, weights_only=False)
    if protein_name in datasets.keys():
        proteins = datasets[protein_name]
    else:
        proteins = chain(*datasets.values())
        if protein_name =='all':
            proteins = list(proteins)

    for protein in proteins:
        print(f'**********************Current dataset: {protein["name"]}**********************')
        # calculate zeroshot score and mutant probability matrix
        print(f'Calculating zeroshot score and mutant probability matrix')
        print(f'Constructing mutant library')
        output_csv_seq = f'{work_dir_base}/{protein["name"]}/{protein["name"]}.csv'
        output_csv_score = f'{work_dir_base}/{protein["name"]}/{protein["name"]}_zeroshot.csv'
        output_npy = f'{work_dir_base}/{protein["name"]}/{protein["name"]}_condition_prob.npy'
        output_fasta = f'{work_dir_base}/{protein["name"]}/{protein["name"]}.fasta'
        seq = protein['wild_type']
        #print(protein['df'])
        
        if os.path.exists(output_csv_seq) and os.path.exists(output_npy) and os.path.exists(output_csv_score) and os.path.exists(output_fasta):
            print(f'Dataset: {protein["name"]} already finished, skipping')
        else:
            labels_seq, labels_score, prob_matrix, all_records = zeroshot_predicting(model_path, seq, protein['df'], device)
            labels_seq.to_csv(output_csv_seq, index=False)
            labels_score.to_csv(output_csv_score, index=False)
            with open(output_fasta, "w") as handle:
                SeqIO.write(all_records, handle, "fasta")
            np.save(output_npy, prob_matrix)

        # extract alpha carbon coordinates
        print(f'Extracting alpha carbon coordinates')
        output_coord = f'{work_dir_base}/{protein["name"]}/{protein["name"]}_ca_coords.npy'
        if os.path.exists(output_coord):
            print(f'Dataset: {protein["name"]} already finished, skipping')
        else:
            coord = extract_alpha_coords(protein['name'], seq)
            np.save(output_coord, coord)

        #extract ProteinMPNN probs
        print(f'Extracting ProteinMPNN probs')
        output_proteinmpnn = f'{work_dir_base}/{protein["name"]}/{protein["name"]}_proteinmpnn_probs.npy'
        if os.path.exists(output_proteinmpnn):
            print(f'Dataset: {protein["name"]} already finished, skipping')
        else:
            proteinmpnn_probs = extract_ProteinMPNN_probs(protein['name'], seq)
            np.save(output_proteinmpnn, proteinmpnn_probs)        
        

if __name__ == "__main__":
    generate_aux_data('all')