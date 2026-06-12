import numpy as np
import torch
from itertools import chain
from pathlib import Path
import subprocess

path = f'/data1/users/weig03/data/Focus_work/EvolvePro-main/data/dms/pdb'
pdb_file_dir = ['brenan.pdb', 'cov2_S.pdb', 'giacomelli.pdb', 'jones.pdb', 'lee.pdb', 'stiffler.pdb',
                'cas12f.pdb', 'doud.pdb', 'haddox.pdb', 'kelsic.pdb', 'markin.pdb', 'zikv_E.pdb']

model_weights_path = 'vanilla_model_weights'  # 请根据实际路径修改
model_name = 'v_48_020'
output_base = Path('output_alde')  # 输出文件夹

for pdb_file in pdb_file_dir:
    print(f'Extracting protein: {pdb_file}')
    protein_name = pdb_file.strip().split('.')[0]
    pdb_path = Path(f"{path}/{pdb_file}")

    # 检查 PDB 文件是否存在
    if not pdb_path.exists():
        print(f'PDB file not found: {pdb_path}')
        continue

    # 设置输出文件夹
    output_dir = output_base / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已处理
    output_file = output_dir / 'conditional_probs_only' / f'{protein_name}.npz'
    if output_file.exists():
        print(f'Output already exists for {protein_name}, skipping.')
        continue

     # 构建命令
    cmd = [
        'python', 'protein_mpnn_run.py',
        '--pdb_path', str(pdb_path),
        '--out_folder', str(output_dir),
        '--model_name', model_name,
        '--path_to_model_weights', model_weights_path,
        '--save_probs', '1',
        '--conditional_probs_only', '1',
        '--num_seq_per_target', '10',
        '--sampling_temp', '0.1',
        '--batch_size', '1'
    ]

    # 调用命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error processing {protein_name}: {e}')
