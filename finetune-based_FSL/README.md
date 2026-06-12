# Finetune-based FSL User Guide

conda activate isde-finetune

## Part 1. Reproducing Finetune-based FSL Baselines (Chapter 1 & Chapter 2)

### Step 1. Dataset Preparation

Place all ProteinGym datasets under:

```bash
ISDE/finetune-based_FSL/data/substitutions/
```

Run the preprocessing script:

```bash
python preprocess.py -s
```

This script preprocesses the raw ProteinGym datasets and packs them into:

```text
data/merged.pkl
```

For detailed instructions on dataset preparation and preprocessing, refer to:

https://github.com/ai4protein/VenusFSFP

---

### Step 2. Model Training and Evaluation

Run:

```bash
bash run.sh
```

This script reproduces the finetune-based FSL baseline experiments reported in Chapter 1 and Chapter 2.

---

# Part 2. Reproducing Finetune-based FSL Experiments (Chapter 3)

## Step 1. Dataset Preparation

Place the following datasets together under:

```bash
ISDE/finetune-based_FSL/data/substitutions/
```

- ProteinGym datasets used in Part 1
- 12 DMS datasets from EVOLVEpro

Then run:

```bash
python preprocess.py -s -sp _AL
```

This script preprocesses the datasets and generates:

```text
data/merged_AL.pkl
```

Preprocessed 12 DMS datasets from EVOLVEpro are available in the Supplementary Files.

---

## Step 2. Active Learning Evaluation

Run:

```bash
python directed_evolution.py
```

This script reproduces the active learning experiments reported in Chapter 3.

---

# Part 3. ISDE Single-site Mutant Prediction Module (Chapter 5)

This module predicts the effects of single-site mutants under the active learning framework.

## Step 1. Prepare Target Protein Sequence

Prepare the FASTA file of the target protein and place it under:

```bash
ISDE/finetune-based_FSL/exp/seq/
```

File organization:

```text
exp/seq/{target_protein}.fa
```

Example:

```text
exp/seq/RsCas12f.fa
```

All RsCas12f-related files can be obtained from the Supplementary Files.

---

## Step 2. Generate Structure-aware Sequence (SaProt Only)

If SaProt is used as the base model, convert the target protein structure into a structure-aware sequence.

Append the generated sequence to:

```bash
ISDE/finetune-based_FSL/data/struc_sq_72.csv
```

For detailed instructions, refer to:

https://github.com/westlake-repl/SaProt

---

## Step 3. Generate GEMME Zero-shot Predictions

Run GEMME on the target protein and obtain the output file:

```text
normPred_evolCombi.txt
```

Place the file under:

```bash
ISDE/finetune-based_FSL/exp/GEMME/{target_protein}/
```

Example:

```text
ISDE/finetune-based_FSL/exp/GEMME/RsCas12f/normPred_evolCombi.txt
```

For more information about GEMME, refer to:

http://www.lcqb.upmc.fr/GEMME

---

## Step 4. Prepare Experimental Data from Previous Rounds

Place all experimental results generated in previous ISDE rounds under:

```bash
ISDE/finetune-based_FSL/exp/round/
```

Example:

```text
{target_protein}_Round1.xlsx
{target_protein}_Round2.xlsx
{target_protein}_Round3.xlsx
...
```

For example:

```text
RsCas12f_Round1.xlsx
RsCas12f_Round2.xlsx
```

---

## Step 5. Predict Single-site Mutant Effects

Open:

```bash
directed_evolution_experimental.py
```

Modify the parameters in the function call at the bottom of the script to ensure all file paths and settings are correct.

Then run:

```bash
python directed_evolution_experimental.py
```
