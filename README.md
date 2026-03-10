# ALS Perturbation Analysis

Pipeline for analyzing ALS perturbation effects in single-cell embeddings.

## Project Layout

- Data: [helical_challenge/counts_combined_filtered_BA4_sALS_PN.h5ad](als_perturbation/counts_combined_filtered_BA4_sALS_PN.h5ad)
- Source code: [als_perturbation/src](als_perturbation/src)
- Notebooks:
  - [helical_challenge/notebooks/01_perturbation_workflow.ipynb](als_perturbation/notebooks/01_perturbation_workflow.ipynb)
  - [helical_challenge/notebooks/02_als_perturbations_geneformer.ipynb](als_perturbation/notebooks/02_als_perturbations_geneformer.ipynb)
  - [helical_challenge/notebooks/03_embedding_interpretation.ipynb](als_perturbation/notebooks/03_embedding_interpretation.ipynb)
  - [als_perthelical_challengeurbation/notebooks/04_target_prioritization.ipynb](als_perturbation/notebooks/04_target_prioritization.ipynb)
- Outputs: [helical_challenge/outputs](als_perturbation/outputs)
- Figures: [helical_challenge/figures](als_perturbation/figures)

## Data

Primary dataset:
- [`counts_combined_filtered_BA4_sALS_PN.h5ad`](als_perturbation/counts_combined_filtered_BA4_sALS_PN.h5ad)

Expected contents (AnnData format):
- `X`: filtered gene expression matrix
- `obs`: per-cell metadata (e.g., condition, cell type, perturbation labels)
- `var`: per-gene metadata
- optional embeddings in `obsm` (if generated during workflow)

Biological scope:
- BA4 motor cortex samples
- sALS and control/healthy context
- PN-focused analysis for perturbation-driven state shifts

## Geneformer Model

This project uses **Geneformer**, a transformer-based foundation model for single-cell transcriptomics, to:
- encode cells into latent embeddings
- simulate/score perturbation effects in embedding space
- compare ALS perturbed states against healthy PN reference states

## Environment Setup

```bash
cd helical_challenge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core Modules

### `src/perturbation.py`
Perturbation simulation and scoring logic:
- Gene knockout/knockdown simulation
- Embedding-space perturbation effects
- Target scoring and prioritization utilities

### `src/analysis.py`
Post-perturbation analysis and metrics:
- `embeddings_to_adata`: build AnnData from embedding outputs and metadata
- `compute_shifts_both_metrics`: perturbation shifts using cosine + euclidean metrics
- `shifts_by_cell_type`: shift analysis stratified by cell type
- `compute_knn_overlap_adata`: mean kNN cosine distance of ALS perturbed cells to healthy PN baseline
- `build_summary_table`: aggregate final prioritization metrics

## Notebooks

### 01_perturbation_workflow.ipynb
Preprocessing and baseline embedding setup:
- Load and explore BA4 sALS + control gene expression data
- Filter cells by type and quality metrics
- Generate initial embeddings and establish healthy PN baseline reference states
- Prepare data for downstream perturbation experiments

### 02_als_perturbations_geneformer.ipynb
Core perturbation analysis using Geneformer:
- Initialize Geneformer model and tokenizer
- Simulate gene perturbations (knockdowns/knockouts) on ALS cells
- Compute embedding shifts for each perturbation
- Score perturbations by movement toward healthy PN baseline
- Generate per-gene and per-cell-type perturbation metrics

### 03_embedding_interpretation.ipynb
Embedding space analysis and visualization:
- Visualize baseline and perturbed cell states in embedding space
- Analyze kNN neighborhoods and topology changes
- Interpret which genes drive cells toward (or away from) healthy states
- Generate summary plots and diagnostic figures

### 04_target_prioritization.ipynb
Final target gene ranking and validation:
- Aggregate perturbation metrics across cell types and distance metrics
- Build summary ranking table of top therapeutic targets
- Validate rankings against domain knowledge and literature
- Export final `summary_table_task3.csv` with target scores

## Suggested Workflow

1. Run [01_perturbation_workflow.ipynb](als_perturbation/notebooks/01_perturbation_workflow.ipynb) to preprocess data and establish baseline embeddings.
2. Run [02_als_perturbations_geneformer.ipynb](als_perturbation/notebooks/02_als_perturbations_geneformer.ipynb) to simulate perturbations and compute shifts.
3. Run [03_embedding_interpretation.ipynb](als_perturbation/notebooks/03_embedding_interpretation.ipynb) to visualize and interpret embedding changes.
4. Run [04_target_prioritization.ipynb](als_perturbation/notebooks/04_target_prioritization.ipynb) to generate final target rankings.

## Outputs

- Summary table: [als_perturbation/outputs/summary_table_task3.csv](als_perturbation/outputs/summary_table_task3.csv)
- Intermediate task outputs: [als_perturbation/outputs/task1](als_perturbation/outputs/task1), [als_perturbation/outputs/task2](als_perturbation/outputs/task2)

## Source

- Geneformer paper: https://www.nature.com/articles/s41586-023-06139-9
- Helical documentation: https://helical.readthedocs.io/en/latest/ 