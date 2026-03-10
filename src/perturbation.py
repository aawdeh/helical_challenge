from __future__ import annotations

import logging
from statistics import mode
import time
from typing import List, Literal, Optional, Dict
from xml.parsers.expat import model

import scipy.sparse as sp
import numpy as np
import anndata as ad
import torch
from helical.models.geneformer import Geneformer

logger = logging.getLogger(__name__)

# ────────────────────────── Private functions ──────────────────────────────────────

def _get_valid_gene_indices(adata: ad.AnnData, genes: List[str]) -> Dict[str, int]:
    """
    Validate genes against adata.var_names and return a mapping
    of {gene_name: column_index} for all found genes.

    Parameters
    ----------
    adata : AnnData object
    genes : list of gene names to look up

    Returns
    -------
    Dict mapping valid gene names to their column indices

    Raises
    ------
    ValueError : if no valid genes are found at all
    """
    if not genes:
        raise ValueError("Gene list is empty.")

    gene_index_map = {}
    for gene in genes:
        if gene in adata.var_names:
            gene_index_map[gene] = adata.var_names.get_loc(gene)
        else:
            logger.warning("Gene '%s' not found in dataset. Skipping.", gene)

    if not gene_index_map:
        raise ValueError(
            f"None of the requested genes {genes} were found in the dataset."
        )

    return gene_index_map

def _modify_sparse_columns(X: sp.spmatrix, col_indices: List[int], value: float) -> sp.csr_matrix:
    """
    Set specified columns of a sparse matrix to a scalar value.
    Converts to CSC for efficient column operations, modifies in place, then converts back to CSR.

    Parameters
    ----------
    X           : input sparse matrix (not modified in place)
    col_indices : list of column indices to modify
    value       : scalar value to assign

    Returns
    -------
    Modified matrix in CSR format
    """
    X_csc = X.tocsc()
    X_csc[:, col_indices] = value
    return X_csc.tocsr()

def _subsample(adata: ad.AnnData, n_cells: int) -> ad.AnnData:
    """
    Randomly subsample n_cells from the AnnData object for faster testing.

    Parameters
    ----------
    adata : AnnData object to subsample
    n_cells : number of cells to sample

    Returns
    -------
    New AnnData object containing only the sampled cells.
    """
    if n_cells >= adata.n_obs:
        logger.warning("Requested cell_subset (%d) exceeds total cells (%d). Using all cells.", n_cells, adata.n_obs)
        return adata

    sampled_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
    return adata[sampled_indices].copy()

def _subsample_als(
    adata: ad.AnnData,
    n_cells: int = 750,
    cell_type_col: str = "CellType",
    random_seed: int = 42,
) -> ad.AnnData:
    """
    Subsample cells with equal representation per cell type.

    Filters to cells expressing enough ALS genes, then samples
    an equal number of cells from each cell type present.

    Parameters
    ----------
    adata          : AnnData object (ALS or healthy — passed separately)
    gene_list      : List of ALS-associated gene names
    n_cells        : Total target cells — split equally across cell types
    cell_type_col  : adata.obs column name for cell type
    min_als_genes  : Minimum ALS genes a cell must express to be kept
    random_seed    : Random seed for reproducibility

    Returns
    -------
    Subsampled AnnData with equal cells per cell type
    """
    np.random.seed(random_seed)

    print(f"Original: {adata.n_obs} cells × {adata.n_vars} genes")
    cell_type_counts = adata.obs[cell_type_col].value_counts()
    n_cell_types     = len(cell_type_counts)
    n_per_type       = max(1, n_cells // n_cell_types)
    print(f"Subsampling to {n_cells} cells total → {n_per_type} per cell type across {n_cell_types} types:\n")

    # Sample cell barcodes from each type
    selected_barcodes = []
    for cell_type, count in cell_type_counts.items():
        ct_barcodes = adata.obs_names[adata.obs[cell_type_col] == cell_type]
        n_sample    = min(n_per_type, len(ct_barcodes))
        sampled     = np.random.choice(ct_barcodes, n_sample, replace=False)
        selected_barcodes.extend(sampled)

    # ── Reindex into original adata — preserves ALL obs and var exactly ───
    adata_final = adata[selected_barcodes, :].copy()

    # ── Verify nothing lost ───────────────────────────────────────────────
    assert set(adata.obs.columns) == set(adata_final.obs.columns), \
        "obs columns mismatch!"
    assert set(adata.var.columns) == set(adata_final.var.columns), \
        "var columns mismatch!"

    print(f"\nFinal : {adata_final.n_obs} cells × {adata_final.n_vars} genes")
    print(f"obs   : {adata_final.obs.columns.tolist()}")
    print(f"var   : {adata_final.var.columns.tolist()}")

    return adata_final

def _validate_pipeline_inputs(adata, genes, modes, model) -> None:
    """
    Validate inputs for the perturbation pipeline. Raises ValueError with descriptive messages if any checks fail.
    """
    if not adata or adata.n_obs == 0:
        raise ValueError("adata is empty or None.")
    if not genes:
        raise ValueError("Gene list is empty.")
    if not modes:
        raise ValueError("Modes list is empty.")
    invalid = set(modes) - {"knockdown", "knockup"}
    if invalid:
        raise ValueError(f"Invalid modes: {invalid}")
    if model is None:
        raise ValueError("Model is None.")

def _clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def compute_embeddings(model, adata):
    """
    Compute embeddings for the given AnnData object using the specified model.

    Parameters
    ----------
    model : Geneformer model instance
    adata : AnnData object

    Returns
    -------
    Embeddings 
    """
    torch.cuda.empty_cache()
    dataset = model.process_data(adata)
    with torch.no_grad():
            embeddings = model.get_embeddings(dataset)
    return embeddings

def _clean_adata(adata: ad.AnnData) -> ad.AnnData:
    """Remove layers and uns to keep AnnData clean."""
    adata = adata.copy()
    adata.layers = {}
    adata.uns    = {}
    return adata

# ────────────────────────── Public functions ──────────────────────────────────────
def generate_perturbations_multiple(
    adata: ad.AnnData,
    gene_list: List[str],
    modes: List[Literal["knockdown", "knockup"]],
    strength: float = 1.0,
    heterogeneity: float = 0.1,
    random_state: int = 42,
    multi_gene_label: Optional[str] = None,
) -> Dict[str, ad.AnnData]:
    """
    Apply gradual stochastic perturbation — single-gene and/or multi-gene.

    Single-gene mode (default): one condition per gene, e.g. "knockdown_SOD1".
    Multi-gene mode: one condition where ALL genes in gene_list are perturbed
                     simultaneously, labelled "{mode}_MULTI_{label}".

    X'_{i,g} = X_{i,g} * exp(delta_i)
    delta_i ~ Normal(mu, heterogeneity^2)

    mu = +strength (knockup)
    mu = -strength (knockdown)

    Parameters
    ----------
    adata            : AnnData with raw counts in .X and gene names in adata.var_names
    gene_list        : list of gene names to perturb (must be in adata.var_names)
    modes            : list of perturbation modes ("knockdown" or "knockup")
    strength         : average log-fold change to apply
    heterogeneity    : std dev of log-fold change across cells
    random_state     : seed for reproducibility
    multi_gene_label : if set, ALSO creates one combined condition where all genes in
                       gene_list are perturbed together, labelled
                       "{mode}_MULTI_{multi_gene_label}".
                       If None, only single-gene conditions are produced.

    Returns
    -------
    Dict mapping condition labels to perturbed AnnData objects.
    Single-gene keys  : "{mode}_{gene_name}"  e.g. "knockdown_SOD1"
    Multi-gene keys   : "{mode}_MULTI_{label}" e.g. "knockdown_MULTI_ALS_genes"
    """
    rng = np.random.default_rng(random_state)

    # Validate and resolve all genes up front
    gene_index_map = _get_valid_gene_indices(adata, gene_list)

    X_orig = adata.X.astype(np.float32)
    is_sparse = sp.issparse(X_orig)
    if is_sparse:
        X_orig = X_orig.tocsc()

    # ── Helper: apply perturbation to one or more columns ─────────────────
    def _perturb(X, col_indices: List[int], mu: float) -> sp.csr_matrix | np.ndarray:
        """Perturb each column in col_indices independently with its own delta draw."""
        X = X.copy()
        for col_idx in col_indices:
            delta = rng.normal(loc=mu, scale=heterogeneity, size=adata.n_obs)
            scaling = np.exp(delta).astype(np.float32)
            if is_sparse:
                col_dense = X[:, col_idx].toarray().ravel()
                X[:, col_idx] = np.round(col_dense * scaling).reshape(-1, 1)
            else:
                X[:, col_idx] = np.round(X[:, col_idx] * scaling)
        return X

    perturbations: Dict[str, ad.AnnData] = {}
    all_col_indices = list(gene_index_map.values())

    # Count total conditions for progress logging
    n_single = len(gene_index_map) * len(modes)
    n_multi  = len(modes) if multi_gene_label else 0
    total_conditions = n_single + n_multi
    condition_counter = 0

    for mode in modes:
        mu = strength if mode == "knockup" else -strength

        # ── Single-gene conditions ─────────────────────────────────────────
        for gene_name, col_idx in gene_index_map.items():
            condition_counter += 1
            label = f"{mode}_{gene_name}"
            logger.info("[%d/%d] Running %s", condition_counter, total_conditions, label)

            X_perturbed = _perturb(X_orig, [col_idx], mu)

            adata_out = adata.copy()
            adata_out.X = (X_perturbed.tocsr() if is_sparse else X_perturbed).astype(np.int32)
            perturbations[label] = adata_out

            logger.info(
                "  ✓ %s | gene=%s | col_idx=%d | strength=%.2f | heterogeneity=%.2f",
                label, gene_name, col_idx, strength, heterogeneity,
            )

        # ── Multi-gene condition (all genes perturbed simultaneously) ──────
        if multi_gene_label:
            condition_counter += 1
            label = f"{mode}_MULTI_{multi_gene_label}"
            logger.info("[%d/%d] Running %s (%d genes)", condition_counter, total_conditions, label, len(all_col_indices))

            X_perturbed = _perturb(X_orig, all_col_indices, mu)

            adata_out = adata.copy()
            adata_out.X = (X_perturbed.tocsr() if is_sparse else X_perturbed).astype(np.int32)
            perturbations[label] = adata_out

            logger.info(
                "  ✓ %s | genes=%s | strength=%.2f | heterogeneity=%.2f",
                label, list(gene_index_map.keys()), strength, heterogeneity,
            )

    return perturbations

def generate_perturbations(
    adata: ad.AnnData,
    gene_list: List[str],
    modes: List[Literal["knockdown", "knockup"]],
    strength: float = 1.0,
    heterogeneity: float = 0.1,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Apply gradual stochastic perturbation.

    X'_{i,g} = X_{i,g} * exp(delta_i)
    delta_i ~ Normal(mu, heterogeneity^2)

    mu = +strength (knockup)
    mu = -strength (knockdown)

    Parameters
    ----------
    adata          : AnnData with raw counts in .X and gene names in adata  
    gene_list      : list of gene names to perturb (must be in adata.var[gene_name_col])
    modes          : list of perturbation modes ("knockdown" or "knockup")
    strength       : average log-fold change to apply (positive for knockup, negative for knockdown)
    heterogeneity  : standard deviation of log-fold change across cells (default 0.1, higher = more variability in response)
    random_state   : seed for reproducibility (default 42)

    Returns
    -------
    Dict mapping condition labels (e.g. "knockdown_SOD1") to their corresponding perturbed AnnData objects.    
    """

    rng = np.random.default_rng(random_state) 

    # Validated {gene_name: col_index} mapping — preserves name↔index correspondence
    gene_index_map = _get_valid_gene_indices(adata, gene_list)

    total_conditions = len(gene_index_map) * len(modes)
    condition_counter = 0
    perturbations = {}
  
    # Cache the original matrix once; never mutate it
    X_orig = adata.X.astype(np.float32)
    is_sparse = sp.issparse(X_orig)
    if is_sparse:
        X_orig = X_orig.tocsc()  # CSC: efficient column slicing, set once

    for mode in modes:
        mu = strength if mode == "knockup" else -strength
        for gene_name, col_idx in gene_index_map.items():
            condition_counter += 1
            condition_label = f"{mode}_{gene_name}"
            logger.info("[%d/%d] Running %s", condition_counter, total_conditions, condition_label)

            # Fresh copy of original matrix per condition — perturbations are independent
            X = X_orig.copy()
            
            # Per-cell log-fold change → multiplicative scaling factor
            delta = rng.normal(loc=mu, scale=heterogeneity, size=adata.n_obs)
            scaling = np.exp(delta).astype(np.float32)

            if is_sparse:
                col_dense = X[:, col_idx].toarray().ravel()
                X[:, col_idx] = np.round(col_dense * scaling).reshape(-1, 1)
            else:
                X[:, col_idx] = np.round(X[:, col_idx] * scaling)

            adata_out = adata.copy()
            adata_out.X = (X.tocsr() if is_sparse else X).astype(np.int32)
            perturbations[condition_label] = adata_out

            logger.info(
                "  ✓ %s | gene=%s | col_idx=%d | strength=%.2f | heterogeneity=%.2f",
                condition_label, gene_name, col_idx, strength, heterogeneity,
            )

    return perturbations

def run_perturbation_pipeline(
    adata: ad.AnnData,
    gene_list: List[str],
    modes: List[str],
    model: Geneformer,
    cell_subset: Optional[int] = None,
    strength: float = 5.0,
    heterogeneity: float = 0.1,
    random_state: int = 42
    ) -> Dict[str, np.ndarray]:

    """
    Run GeneFormer embeddings for baseline + all gene × perturbation mode combinations.
    Gradual perturbation is applied to simulate more realistic biological responses, 
    where not all cells respond identically to a perturbation.
    By randomly sampling a log-fold change for each cell from a normal distribution, 
    we can create a range of responses to the perturbation across the cell population. 
    This allows us to capture the heterogeneity in how different cells might react to the same genetic perturbation, 
    which is more reflective of real biological systems where not all cells respond uniformly.
    
    Parameters
    ----------
    adata          : AnnData with raw counts in .X and gene names in adata.var[gene_name_col]
    gene_list      : list of gene names to perturb (must be in adata.var[gene_name_col])
    modes          : list of perturbation modes to apply (e.g. ["knockdown", "knockup"])
    model          : pre-initialized Geneformer model (already loaded on correct device)
    cell_subset    : if set, randomly sample this many cells from adata for faster testing (default None = use all cells)
    strength       : strength of perturbation to apply (default 0.5)
    heterogeneity  : variability of perturbation across cells (default 0.1)
    random_state   : seed for reproducibility (default 42)

    Returns
    -------
    Dict mapping condition labels (e.g. "knockdown_SOD1") to their corresponding embedding arrays.
    """

    _validate_pipeline_inputs(adata, gene_list, modes, model)

    if cell_subset:
        adata = _subsample(adata, cell_subset)

    logger.info("Starting perturbation pipeline.")
    logger.info("Genes: %s", gene_list)
    logger.info("Modes: %s", modes)

    # Results dictionary to hold embeddings for baseline and all perturbations
    results: Dict[str, np.ndarray] = {}
    labels = adata.obs["CellType"]
    conditions = adata.obs["Condition"]
    
    # Baseline Embeddings
    logger.info("Computing baseline embeddings...")
    results["baseline"] = compute_embeddings(model, adata)

    # Perturbations
    results["perturbations"] = {}
    perturbations = generate_perturbations(
                        adata,
                        gene_list,
                        modes,
                        strength=strength,
                        heterogeneity=heterogeneity,
                        random_state=random_state
                    )

    for key, perturbed_adata in perturbations.items():
        logger.info(f"Running perturbation {key}")
        results["perturbations"][key] = compute_embeddings(model, perturbed_adata)

    logger.info("Perturbation pipeline complete. %d conditions embedded.", len(results))
    return results, labels, conditions

def split_by_condition(
    adata: ad.AnnData,
    condition_col: str = "disease",
    healthy_label: str = "PN",
    disease_label: str = "ALS",
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Split AnnData into healthy and disease subsets.

    Parameters
    ----------
    adata           : full AnnData object
    condition_col   : obs column containing disease status
    healthy_label   : label for healthy/control cells
    disease_label   : label for disease cells

    Returns
    -------
    (adata_healthy, adata_disease)
    """
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Column '{condition_col}' not found in adata.obs")

    adata_healthy = adata[adata.obs[condition_col] == healthy_label].copy()
    adata_disease = adata[adata.obs[condition_col] == disease_label].copy()

    logger.info(
        "Split: %d healthy (%s) | %d disease (%s)",
        adata_healthy.n_obs, healthy_label,
        adata_disease.n_obs, disease_label,
    )
    return adata_healthy, adata_disease