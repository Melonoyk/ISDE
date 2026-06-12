import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os

def extract_alpha_coords(dataset: str) -> None:
    """提取并验证PDB与FASTA序列后保存α碳坐标"""
    # 构造文件路径
    base_dir = Path("data/dms")
    pdb_path = base_dir / "pdb" / f"{dataset}.pdb"
    fasta_path = base_dir / "wt_fasta" / f"{dataset}_WT.fasta"
    output_dir = Path("output/dms")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 读取FASTA序列
    with open(fasta_path) as handle:
        fasta_seq = str(next(SeqIO.parse(handle, "fasta")).seq)
    
    # 解析PDB文件
    parser = PDBParser()
    structure = parser.get_structure(dataset, pdb_path)
    
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
            f"Sequence mismatch at dataset:{dataset}!\nFASTA: {fasta_seq[:10]}...\nPDB: {pdb_seq[:10]}..."
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
    output_path = output_dir / f"{dataset}_ca_coords.npy"
    np.save(output_path, np.array(ca_coords))
    print(f"Saved {len(ca_coords)} CA coordinates to {output_path}")

if __name__ == "__main__":
    #datasets=("brenan" "stiffler" "doud" "haddox" "giacomelli" "jones" "kelsic" "lee" "markin" "zikv_E" "cas12f" "cov2_S")
    datasets=["brenan", "stiffler", "doud", "haddox", "giacomelli", "jones", "kelsic", "lee", "markin", "cas12f", "cov2_S", "zikv_E"]
    for dataset in datasets:
        print(f'Processing {dataset}...')
        extract_alpha_coords(dataset)
