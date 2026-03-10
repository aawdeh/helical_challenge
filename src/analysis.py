from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# AnnData builder 
def embeddings_to_adata(results: Dict[str, np.ndarray], cells: pd.DataFrame, conditions: pd.DataFrame) -> ad.AnnData:
    """
    Convert a perturbation results dict into an AnnData object.

    Stacks all condition embeddings row-wise so scanpy tools
    (neighbors, umap, leiden) can operate on them directly.

    Parameters
    ----------
    results : {"baseline": arr, "knockdown_TARDBP": arr, ...}
    cells   : DataFrame with cell type annotations
    conditions : DataFrame with condition annotations (e.g. "PN" vs "ALS")

    Returns
    -------
    AnnData where:
        .X          = stacked embeddings (n_total_cells, embedding_dim)
        .obs["condition"] = condition label per cell
        .obs["gene"]      = gene name per cell
        .obs["mode"]      = perturbation mode per cell
        .obs["cell_type"] = cell type annotation per cell
        .obs["disease"]   = disease vs healthy annotation per cell (PN vs ALS)
            
    """
    arrays, labels, genes, modes, cell_types, diseases = [], [], [], [], [], []

    # Baseline first 
    arrays.append(results["baseline"])
    labels.extend(["baseline"] * len(results["baseline"]))
    genes.extend(["baseline"] * len(results["baseline"]))
    modes.extend(["baseline"] * len(results["baseline"]))
    cell_types.extend(cells["CellType"].tolist())
    diseases.extend(conditions["Condition"].tolist())

    for label, emb in results["perturbations"].item().items():
        arrays.append(emb)
        labels.extend([label] * len(emb))
        mode, gene = label.split("_", 1)
        genes.extend([gene] * len(emb))
        modes.extend([mode] * len(emb))
        cell_types.extend(cells["CellType"].tolist())
        diseases.extend(conditions["Condition"].tolist())

    adata = ad.AnnData(X=np.vstack(arrays))
    adata.obs["condition"] = labels
    adata.obs["gene"]      = genes
    adata.obs["mode"]      = modes
    adata.obs["cell_type"] = cell_types
    adata.obs["disease"]   = diseases
    # Cast to categorical for scanpy plotting
    for col in ["condition", "gene", "mode", "cell_type", "disease"]:
        adata.obs[col] = adata.obs[col].astype("category")

    logger.info("Built AnnData: %d cells, %d dims", adata.n_obs, adata.n_vars)
    return adata

# Cosine + L2 shift 
def compute_shifts_both_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """
    Compute cosine and L2 distance from baseline centroid
    for each perturbed condition.

    L2 (Euclidean) measures the straight-line distance between two points in space. 
    It cares about both direction and magnitude.
    
    Cosine distance only measures the angle between two vectors. It ignores magnitude entirely.

    Parameters
    ----------
    adata : AnnData with obs columns: condition, gene, mode
            and .X containing embeddings (n_cells, 512)

    Returns
    -------
    DataFrame sorted by cosine_shift descending:
        gene | mode | cosine_shift | l2_shift
    """
    # Extract baseline embeddings
    baseline = adata[adata.obs["mode"] == "baseline"].X
    if sp.issparse(baseline):
        baseline = baseline.toarray()
    
    # Compute baseline centroid (mean embedding across all baseline cells)
    baseline_centroid = baseline.mean(axis=0, keepdims=True)

    # For each perturbed condition, compute cosine and L2 distance to baseline centroid
    rows = []
    for condition in adata.obs["condition"].unique():
        if condition == "baseline":
            continue

        mode, gene = condition.split("_", 1)
        emb = adata[adata.obs["condition"] == condition].X
        if sp.issparse(emb):
            emb = emb.toarray()
        centroid = emb.mean(axis=0, keepdims=True)

        rows.append({
            "gene":         gene,
            "mode":         mode,
            "cosine_shift": round(cosine_distances(baseline_centroid, centroid)[0, 0], 6),
            "l2_shift":     round(euclidean_distances(baseline_centroid, centroid)[0, 0], 6),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("cosine_shift", ascending=False)
        .reset_index(drop=True)
    )

# Cosine shift by cell type
def shifts_by_cell_type(adata: ad.AnnData) -> pd.DataFrame:
    """
    For each (gene, mode, cell_type) combination, compute cosine
    shift from the baseline centroid of that same cell type.

    This allows us to see if certain perturbations cause bigger shifts in specific cell types, which could indicate cell-type-specific effects.

    Parameters  
    ----------
    adata : AnnData with obs columns: condition, gene, mode, cell_type
            and .X containing embeddings (n_cells, 512)
    
    Returns
    -------
    DataFrame sorted by cosine_shift descending:
        cell_type | gene | mode | cosine_shift

    """
    rows = []
    cell_types = adata.obs["cell_type"].unique()

    for ct in cell_types:
        ct_mask     = adata.obs["cell_type"] == ct
        baseline    = adata[ct_mask & (adata.obs["mode"] == "baseline")].X
        if len(baseline) == 0:
            continue
        baseline_c  = baseline.mean(axis=0, keepdims=True)

        for condition in adata.obs["condition"].unique():
            if condition == "baseline":
                continue
            mode, gene = condition.split("_", 1)
            cond_embs  = adata[ct_mask & (adata.obs["condition"] == condition)].X
            if len(cond_embs) == 0:
                continue
            shift = cosine_distances(baseline_c, cond_embs.mean(axis=0, keepdims=True))[0,0]
            rows.append({"cell_type": ct, "gene": gene, "mode": mode, "cosine_shift": shift})

    return pd.DataFrame(rows)

# kNN neighbourhood analysis
def compute_knn_overlap_adata(adata_all, k=50):
    """
    Takes adata_all with obs columns: condition, gene, mode, disease
    1. Fits kNN on healthy baseline (PN baseline cells)
    2. For each ALS perturbation, measures mean distance to its kNN in the healthy baseline
     - Low distance = perturbation moves ALS cells toward healthy state
     - High distance = perturbation moves ALS cells away from healthy state

    Parameters
    ----------
    adata_all : AnnData with all conditions, obs columns: condition, gene, mode
                    disease (PN vs ALS), and .X with embeddings
    k         : number of neighbors for kNN
    
    Returns
    -------
    DataFrame sorted by mean_knn_dist_to_healthy ascending:
        gene | mode | mean_knn_dist_to_healthy
    """
    # Healthy baseline = PN cells, unperturbed
    mask_baseline = (adata_all.obs["disease"] == "PN") & (adata_all.obs["mode"] == "baseline")
    X_baseline = adata_all[mask_baseline].X

    # Fit kNN on healthy baseline
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X_baseline)

    rows = []
    # ALS perturbed cells only (exclude ALS baseline)
    als_perturbed = adata_all[
        (adata_all.obs["disease"] == "ALS") & 
        (adata_all.obs["mode"] != "baseline")
    ]

    # For each gene + mode, compute mean kNN distance to healthy baseline
    for (gene, mode), group_idx in als_perturbed.obs.groupby(["gene", "mode"]).groups.items():
        X_perturbed = als_perturbed[group_idx].X
        dists, _ = nn.kneighbors(X_perturbed)
        rows.append({
            "gene": gene,
            "mode": mode,
            "mean_knn_dist_to_healthy": dists.mean()
        })

    return pd.DataFrame(rows).sort_values("mean_knn_dist_to_healthy")

def build_summary_table(shifts_disease, shifts_healthy, knn_df):
    """
    Combines cosine shift, selectivity, and kNN distance into a ranked summary table.
    
    Parameters
    ----------
    shifts_disease : pd.DataFrame  — output of shifts_by_cell_type(adata_disease)
    shifts_healthy : pd.DataFrame  — output of shifts_by_cell_type(adata_healthy)
    knn_df         : pd.DataFrame  — output of compute_knn_overlap_adata(adata_all)
    
    Returns
    -------
    pd.DataFrame ranked by overall therapeutic potential
    """
    # Average cosine shift across cell types
    als = shifts_disease.groupby(["gene", "mode"])["cosine_shift"].mean().reset_index()
    als.columns = ["gene", "mode", "cosine_shift_als"]

    pn = shifts_healthy.groupby(["gene", "mode"])["cosine_shift"].mean().reset_index()
    pn.columns = ["gene", "mode", "cosine_shift_pn"]

    # Merge all three sources
    df = als.merge(pn, on=["gene", "mode"]).merge(knn_df, on=["gene", "mode"])

    # Selectivity
    df["selectivity"] = df["cosine_shift_als"] / (df["cosine_shift_pn"] + 1e-10)

    # Rank each metric
    df["knn_rank"]         = df["mean_knn_dist_to_healthy"].rank(ascending=True)
    df["selectivity_rank"] = df["selectivity"].rank(ascending=False)
    df["cosine_rank"]      = df["cosine_shift_als"].rank(ascending=False)
    df["overall_rank"]     = (df["knn_rank"] + df["selectivity_rank"] + df["cosine_rank"]) / 3

    df = df.sort_values("overall_rank")

    # Format for display
    df["cosine_shift_als"]        = (df["cosine_shift_als"] * 1e4).round(3)
    df["cosine_shift_pn"]         = (df["cosine_shift_pn"]  * 1e4).round(3)
    df["selectivity"]             = df["selectivity"].round(2)
    df["mean_knn_dist_to_healthy"] = df["mean_knn_dist_to_healthy"].round(6)
    df["overall_rank"]            = df["overall_rank"].round(1)

    df = df[["gene", "mode", "cosine_shift_als", "cosine_shift_pn",
             "selectivity", "mean_knn_dist_to_healthy", "overall_rank"]]
    df.columns = ["Gene", "Mode", "Cosine shift ALS (×10⁻⁴)", "Cosine shift PN (×10⁻⁴)",
                  "Selectivity", "kNN dist to healthy", "Overall rank"]

    return df