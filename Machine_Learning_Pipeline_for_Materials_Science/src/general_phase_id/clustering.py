"""
clustering.py — Part 2: Unsupervised Clustering with Automatic Phase-Count Selection
======================================================================================

Two approaches:
A) Gaussian Mixture Model (GMM) with BIC-based component selection
B) UMAP dimensionality reduction followed by HDBSCAN

Both return per-atom cluster labels compatible with the output of build_descriptors().
"""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings


def gmm_clustering(
    descriptors: pd.DataFrame,
    min_components: int = 2,
    max_components: int = 9,
    random_state: int = 42
) -> dict:
    """
    Fit GMM for a range of component counts and select best using BIC.

    Parameters
    ----------
    descriptors : pd.DataFrame
        Output from build_general_descriptors(). Must have structure_id and type columns.
        All other columns are treated as features.
    min_components : int
        Minimum number of components to try.
    max_components : int
        Maximum number of components to try (exclusive, so range is [min, max)).
    random_state : int

    Returns
    -------
    dict with keys:
        'labels' : per-atom cluster label (0..k-1) for best model
        'best_n_components' : selected component count
        'bic_table' : pd.DataFrame with n_components and BIC values
        'model' : fitted GMM
        'probs' : per-atom membership probabilities (soft assignment)
    """
    feature_cols = [c for c in descriptors.columns if c not in ['structure_id', 'type']]
    X = descriptors[feature_cols].values

    # Standardize features for GMM (common practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle NaN by dropping rows (or imputing — check count first)
    nan_mask = np.isnan(X_scaled).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"  WARNING: {nan_mask.sum()} rows with NaN values dropped from GMM fitting")
        X_clean = X_scaled[~nan_mask]
    else:
        X_clean = X_scaled

    bic_scores = []
    models = {}
    best_bic = np.inf
    best_model = None
    best_n = min_components

    for n in range(min_components, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=random_state, n_init=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(X_clean)
        bic = gmm.bic(X_clean)
        bic_scores.append((n, bic))
        models[n] = gmm
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_n = n

    bic_table = pd.DataFrame(bic_scores, columns=['n_components', 'BIC'])

    # Assign labels for best model (use full dataset with NaN handling)
    labels = np.full(len(X_scaled), -1, dtype=int)
    probs = np.full((len(X_scaled), best_n), np.nan)
    if nan_mask.sum() > 0:
        labels[~nan_mask] = best_model.predict(X_clean)
        probs[~nan_mask] = best_model.predict_proba(X_clean)
    else:
        labels = best_model.predict(X_clean)
        probs = best_model.predict_proba(X_clean)

    return {
        'labels': labels,
        'best_n_components': best_n,
        'bic_table': bic_table,
        'model': best_model,
        'probs': probs,
    }


def umap_hdbscan_clustering(
    descriptors: pd.DataFrame,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 5,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    random_state: int = 42
) -> dict:
    """
    UMAP dimensionality reduction followed by HDBSCAN clustering.

    Parameters
    ----------
    descriptors : pd.DataFrame
        Output from build_general_descriptors().
    n_neighbors, min_dist, n_components : UMAP parameters
    min_cluster_size, min_samples, cluster_selection_epsilon : HDBSCAN parameters
    random_state : int

    Returns
    -------
    dict with keys:
        'labels' : per-atom cluster label (-1 for noise/outliers)
        'n_clusters' : number of clusters found (excluding noise)
        'noise_fraction' : fraction of atoms labeled as noise (-1)
        'umap_embedding' : 2D UMAP embedding (for visualization)
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        print("ERROR: umap-learn and hdbscan required. Install with:")
        print("  pip install umap-learn hdbscan")
        return {'labels': None, 'n_clusters': 0, 'noise_fraction': 1.0, 'umap_embedding': None}

    feature_cols = [c for c in descriptors.columns if c not in ['structure_id', 'type']]
    X = descriptors[feature_cols].values

    # Handle NaN by dropping rows
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"  WARNING: {nan_mask.sum()} rows with NaN values dropped from UMAP/HDBSCAN")
        X_clean = X[~nan_mask]
    else:
        X_clean = X

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # UMAP reduction
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_scaled)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels_clean = clusterer.fit_predict(embedding)

    # Map back to full dataset
    labels = np.full(len(X), -1, dtype=int)
    if nan_mask.sum() > 0:
        labels[~nan_mask] = labels_clean
    else:
        labels = labels_clean

    n_clusters = len(set(labels_clean)) - (1 if -1 in labels_clean else 0)
    noise_frac = (labels == -1).sum() / len(labels)

    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'noise_fraction': noise_frac,
        'umap_embedding': embedding if nan_mask.sum() == 0 else None,
        'clusterer': clusterer,
    }


def evaluate_clustering(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> dict:
    """
    Evaluate clustering against ground-truth labels using NMI and ARI.

    Parameters
    ----------
    true_labels : array-like
        Ground-truth labels (e.g., 0/1/2 for known compositions, or -1 for unlabeled).
    predicted_labels : array-like
        Cluster labels from clustering method.

    Returns
    -------
    dict with 'nmi', 'ari', plus raw scores.
    """
    # Exclude noise points (-1) from evaluation if present
    valid_mask = (true_labels >= 0) & (predicted_labels >= 0)
    if valid_mask.sum() == 0:
        return {'nmi': np.nan, 'ari': np.nan, 'n_valid': 0}

    n_valid = valid_mask.sum()
    nmi = normalized_mutual_info_score(true_labels[valid_mask], predicted_labels[valid_mask])
    ari = adjusted_rand_score(true_labels[valid_mask], predicted_labels[valid_mask])

    return {
        'nmi': nmi,
        'ari': ari,
        'n_valid': n_valid,
        'total': len(true_labels),
    }