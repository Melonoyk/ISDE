# Top Model-based FSL User Guide

conda activate isde-top_model 

## Part 1. Reproducing Top Model-based FSL Baselines (Chapter 1 & Chapter 2)

### Step 1. Raw Data Preparation

#### 1. Protein Structure Files (PDB)

Place all ProteinGym PDB files under:

```bash
ISDE/top_model-based_FSL/data/proteingym/pdb_proteingym/
```

File organization:

```text
data/proteingym/pdb_proteingym/{protein_name}.pdb
```

#### 2. ProteinMPNN Conditional Probability Matrix

For each protein in ProteinGym, generate conditional probabilities using ProteinMPNN.

The script `extract_mpnnprob_proteingym.py` (used in this work) should be placed under the ProteinMPNN repository.

For ProteinMPNN installation and usage, refer to:

https://github.com/dauparas/ProteinMPNN

#### 3. Copy `merged.pkl`

Copy `merged.pkl` from the finetune-based module:

```bash
cp ../finetune-based_FSL/data/merged.pkl ./data/proteingym/
```

---

### Step 2. Auxiliary Data Extraction

Run:

```bash
python cal_aux_proteingym.py
```

This script extracts:

- Cα coordinates
- ESM2 zero-shot prediction results
- Probability matrices from ESM2 and ProteinMPNN
- Mutant sequences in FASTA format
- Mutant sequences in CSV format

---

### Step 3. PLM Embedding Extraction

Run:

```bash
python extract_emb_proteingym.py
```

This script uses ESM2 to extract embeddings for all mutants generated in the previous step.

---

### Step 4. Benchmark Evaluation

Run:

```bash
python ./gp/pipeline.py
```

This script evaluates all Top Model-based FSL baselines, including:

- Kermut+
- SVM
- Gradient Boosting Tree (GBT)
- Linear Regression
- Random Forest (RF)

---

# Part 2. Reproducing Top Model-based FSL Experiments (Chapter 3)

## Step 1. Raw Data Preparation

### 1. Protein Structure Files (PDB)

Place PDB files of the 12 DMS datasets used in EVOLVEpro under:

```bash
ISDE/top_model-based_FSL/data/dms/pdb/
```

Example:

```text
ISDE/top_model-based_FSL/data/dms/pdb/brenan.pdb
```

All corresponding PDB files can be obtained from the Supplementary Files.

---

### 2. ProteinMPNN Conditional Probability Matrix

Generate ProteinMPNN conditional probabilities for each protein.

The script `extract_mpnnprob_proteingym.py` can be used as an example and should be placed under the ProteinMPNN repository.

ProteinMPNN repository:

https://github.com/dauparas/ProteinMPNN

---

### 3. Other Required Files

Download all raw data from the EVOLVEpro repository and place them under:

```bash
data/dms/
```

Then prepare:

- Mutant sequences in FASTA format
- Mutant sequences in CSV format
- PLM embeddings

following the EVOLVEpro workflow.

---

## Step 2. Data Processing

### ESM2 Zero-shot Predictions

Run:

```bash
python get_zeroshot_mut.py
```

to generate ESM2 zero-shot prediction scores for the 12 DMS datasets.

---

### ESM2 Probability Matrix

Run:

```bash
python get_condition_prob.py
```

to generate conditional probability matrices from ESM2.

---

### ProteinMPNN Probability Matrix

Run:

```bash
python calc_mpnn_prob_alde.py
```

to generate conditional probability matrices from ProteinMPNN.

---

### Cα Coordinates

Run:

```bash
python get_coord.py
```

to extract Cα coordinates.

---

## Step 3. Active Learning Evaluation

Run:

```bash
python ./gp/evolve_r1.py
```

This script evaluates the performance of different acquisition function combinations.

---

# Part 3. ISDE Multi-site Mutant Prediction Module (Chapter 5)

This module predicts the effects of multi-site mutants under the active learning framework.

## Step 1. Raw Data Preparation

### 1. Protein Structure File

Place the target protein PDB file under:

```bash
ISDE/top_model-based_FSL/exp/struc/
```

Example:

```text
ISDE/top_model-based_FSL/exp/struc/RsCas12f.pdb
```

All RsCas12f-related files can be obtained from the Supplementary Files.

---

### 2. ProteinMPNN Output

Run ProteinMPNN on the target protein and obtain the raw ProteinMPNN output files.

---

## Step 2. Mutant Generation and Embedding Extraction

Generate mutant sequences:

```bash
cd ../finetune-based_FSL
python generate_multi_mut.py
```

This will generate:

- FASTA files
- CSV files

for the target protein.

Next, generate PLM embeddings following the EVOLVEpro workflow.

Place all generated files under:

```bash
ISDE/finetune-based_FSL/exp/preprocess/{target_protein}/
```

---

## Step 3. Multi-site Mutation Effect Prediction

Switch to the Top Model-based module:

```bash
cd ../top_model-based_FSL
```

Before running the prediction script:

- Modify the parameters at the bottom of the script.
- Ensure all file paths are correctly configured.

Then run:

```bash
python ./gp/directed_evo_exp.py
```
