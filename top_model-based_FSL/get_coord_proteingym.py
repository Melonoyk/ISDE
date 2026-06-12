import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import torch
from itertools import chain

def extract_alpha_coords(
    dataset_name: str,
    wt_seq: str,
    output_path: str = "output_proteingym/dms/{dataset_name}/{dataset_name}_ca_coords.npy",
) -> None:
    """提取并验证PDB与FASTA序列后保存alpha碳坐标"""
    # 构造文件路径
    parts = dataset_name.strip().split('_')
    protein_name = parts[0] + '_' + parts[1]
    pdb_path = Path(f"data/proteingym/pdb_proteingym/{protein_name}.pdb")
    
    fasta_seq = wt_seq
    
    # 解析PDB文件
    parser = PDBParser()
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
    
    # 序列一致性验证
    if pdb_seq != fasta_seq:
        print(
            f"Sequence mismatch at dataset:{dataset_name}!\nFASTA: {fasta_seq[:10]}...\nPDB: {pdb_seq[:10]}..."
        )
        for i in range(min(len(fasta_seq), len(pdb_seq))):
            if fasta_seq[i] != pdb_seq[i]:
                print(f"Mismatch at position {i+1}: FASTA={fasta_seq[i]}, PDB={pdb_seq[i]}")
        return
    
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
    
    # 保存为numpy数组
    np.save(output_path, np.array(ca_coords))
    print(f"Saved {len(ca_coords)} CA coordinates to {output_path}")

if __name__ == "__main__":
    path = f'data/merged.pkl'
    datasets = torch.load(path, weights_only=False)
    proteins = chain(*datasets.values())
    proteins = list(proteins)
    
    for protein in proteins:
        print(f'**********************Current dataset: {protein["name"]}**********************')
        extract_alpha_coords(protein["name"], protein["wild_type"])
