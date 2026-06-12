#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
import sys
from tqdm import tqdm

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def run(
    work_dir_base,
    model_location,
    toks_per_batch,
    truncation_seq_length=1022,
    repr_layers = [-1]
):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    # get dataset name
    all_datasets = os.listdir(work_dir_base)
        
    for dataset_name in tqdm(all_datasets, desc="Processing datasets"):
        print(f'**********************Current dataset: {dataset_name}**********************')
        # getting output dir
        output_dir = pathlib.Path(
            f"{work_dir_base}/{dataset_name}/pt/"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = f"{output_dir.parent}/{dataset_name}_{model_location}.csv"
        if os.path.exists(output_csv):
            print(f"File {output_csv} already exists. Skipping.")
            continue
        
        fasta_file = os.path.join(work_dir_base, dataset_name, f"{dataset_name}.fasta")
        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
        )
        print(f"Read {fasta_file} with {len(dataset)} sequences")
        
        assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
        repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

        with torch.no_grad():
            valid_name = 1
            for (labels, strs, toks) in tqdm(data_loader, desc="Processing emb from label data"):
                if torch.cuda.is_available():
                    toks = toks.to(device="cuda", non_blocking=True)
                
                out = model(toks, repr_layers=repr_layers, return_contacts=False)

                logits = out["logits"].to(device="cpu")
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    
                    if len(label) <= 200:
                        output_file = output_dir / f"{label}.pt"
                    else:
                        output_file = output_dir / f"{valid_name}.pt"
                        valid_name += 1
                    valid_name += 1
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    result = {"label": label}
                    truncate_len = min(truncation_seq_length, len(strs[i]))
                    # Call clone on tensors to ensure tensors are not views into a larger representation
                    # See https://github.com/pytorch/pytorch/issues/1995
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                    torch.save(
                        result,
                        output_file,
                    )

        print(f"Saved representations to {output_dir}")

        # Concatenate all files in the output directory
        concatenate_files(output_dir, output_csv)


def concatenate_files(output_dir, output_csv):
    # Get all .pt files in the output directory
    files = []
    for r, d, f in os.walk(output_dir):
        for file in f:
            if '.pt' in file:
                files.append(os.path.join(r, file))

    # Load each file and append to a list of dataframes
    dataframes = []
    for file_path in files:
        file_data = torch.load(file_path)
        label = file_data['label']
        representations = file_data['mean_representations']
        key, tensor = representations.popitem()
        row_name = label
        row_data = tensor.tolist()
        new_df = pd.DataFrame([row_data], index=[row_name])
        dataframes.append(new_df)

    # Concatenate all dataframes
    if dataframes:
        concatenated_df = pd.concat(dataframes)
        print("Shape of concatenated DataFrame:", concatenated_df.shape)
        concatenated_df.to_csv(output_csv)
        print(f"Saved concatenated representations to {output_csv}")
    else:
        print("No data to concatenate.")


if __name__ == "__main__":
    run(
        work_dir_base="output_proteingym/dms",
        model_location="esm2_t33_650M_UR50D",
        toks_per_batch=2000,
        truncation_seq_length=1022,
    )