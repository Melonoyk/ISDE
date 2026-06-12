import torch
from itertools import chain
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import sys

def generate_mut_proteingym(
    protein_name: str,
    output_path: str,
):
    path = f'data/merged.pkl'
    datasets = torch.load(path, weights_only=False)
    if protein_name in datasets.keys():
        proteins = datasets[protein_name]
    else:
        proteins = chain(*datasets.values())
        if protein_name =='all':
            proteins = list(proteins)
    
    for protein in proteins:
        print(f'**********************Current dataset: {protein["name"]}**********************')
        output_file = os.path.join(output_path, f'{protein["name"]}',f'{protein["name"]}.fasta')
        if os.path.exists(output_file):
            print(f'{output_file} already exists, skip')
            continue
        #print(protein)
        wt_seq = protein['wild_type']
        dms_data = protein['df']
        print(wt_seq)
        print(dms_data)
        all_records = []
        # 遍历每个突变体
        for mutant_id, row in dms_data.iterrows():
            mutated_sequence = list(wt_seq)  # 转换为列表便于修改
            
            # 解析单突变或多突变（如"Q132I:R79N"）
            mutations = re.split(r":", mutant_id)
            for mutation in mutations:
                # 解析突变位点（示例：R79N → 原始aa=R，位置=79，突变aa=N）
                orig_aa = mutation[0]
                pos = int(mutation[1:-1])
                new_aa = mutation[-1]
                
                # 验证位置有效性
                pos_index = pos - 1  # 转换为0-based索引
                if pos_index >= len(mutated_sequence) or pos_index < 0:
                    print(f"Warning: {mutant_id} position {pos} out of index")
                    break
                # 验证原始氨基酸匹配
                if mutated_sequence[pos_index] != orig_aa:
                    print(f"Warning: {mutant_id} position {pos} should be {orig_aa}, but in wt sequence is {mutated_sequence[pos_index]}")
                    break
                
                # 执行氨基酸替换
                mutated_sequence[pos_index] = new_aa
            
            # 生成SeqRecord对象（参考网页7、8）
            mutated_seq = "".join(mutated_sequence)
            record = SeqRecord(
                Seq(mutated_seq),
                id=mutant_id,
                description=""
            )
            
            all_records.append(record)

        # Write the mutant sequences to a FASTA file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as handle:
            SeqIO.write(all_records, handle, "fasta")
        
        # Print the number of mutants generated
        print(f"Number of mutants: {len(all_records)}")

if __name__ == '__main__':
    generate_mut_proteingym(protein_name='all', output_path='./output_proteingym/dms')