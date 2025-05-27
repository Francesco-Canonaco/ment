
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot
import random
import networkx as nx
from sklearn.metrics import mean_squared_error
import cdt
from cdt.metrics import SHD
from itertools import product



def sample_from_disjoint_interval(size):
    coin_flip = np.random.rand(size) < 0.5
    left = np.random.uniform(-1.5, -0.5, size)
    right = np.random.uniform(0.5, 1.5, size)
    return np.where(coin_flip, left, right)

def generate_connection_matrix(p, s):
    #s is the probability of a connection, if you 
    #want the sparsity to be the same as in the paper 
    #set s=1/(p-1)

    # Generate binary adjacency matrix (lower triangle only)
    rand_matrix = np.random.binomial(1, s, size=(p, p))
    lower_triangle_mask = np.tril(np.ones((p, p)), k=-1)
    adj_matrix = rand_matrix * lower_triangle_mask

    # Count how many 1s (non-zero connections) to sample that many weights
    num_connections = int(np.sum(adj_matrix))

    # Generate random weights from the disjoint interval
    weights = sample_from_disjoint_interval(num_connections)

    # Fill the weights into the matrix (vectorized)
    connection_matrix = np.zeros((p, p))
    connection_matrix[adj_matrix == 1] = weights

    return connection_matrix

def sample_uniform_noise(n, p, variances):
    # Uniform distribution on [-sqrt(3 * var), sqrt(3 * var)] has variance = var
    e = np.zeros((n, p))
    for i in range(p):
        scale = np.sqrt(3 * variances[i])
        e[:, i] = np.random.uniform(low=-scale, high=scale, size=n)
    return e

def generate_dataset(E, B, n, permutation=None):
    """
    Generate dataset X from external influences E and connection matrix B,
    with Gaussian shift and optional column permutation.
    
    Parameters:
    - E: (n, p) matrix of external influences
    - B: (p, p) connection strength matrix
    - permutation: list or array of length p (optional). If None, a random permutation is generated.
    
    Returns:
    - X_perm: the permuted data matrix (n, p)
    - means: Gaussian shifts applied to each variable (p,)
    - permutation: the permutation used (to apply consistently across groups)
    """
    n, p = E.shape
    I = np.eye(p)
    A_inv = np.linalg.inv(I - B).T
    X = E @ A_inv  # Step: X = E (I - B)^-1

    # Step: Add Gaussian mean shift (N(0, 4) → std = 2)
    means = np.random.normal(loc=0, scale=2.0, size=p)
    X += means  # Broadcast to shift each variable

    # Step: Apply consistent variable permutation
    if permutation is None:
        permutation = np.random.permutation(p)

    X_perm = X[:, permutation]  # Permute columns
    
    return X_perm, means, permutation




def perturb_adjacency_matrix(matrix, n=1, mode='both'):
    """
    Perturbs a weighted adjacency matrix by changing existing weights and 
    optionally adding/removing edges, preserving causal order (j < i).

    Parameters:
    - matrix: np.ndarray, square weighted adjacency matrix
    - n: int, number of edges to add/remove
    - mode: str, one of 'add', 'remove', or 'both'

    Returns:
    - new_matrix: np.ndarray, perturbed matrix
    """
    new_matrix = matrix.copy()
    num_nodes = new_matrix.shape[0]
    
    # Get lower triangle indices (j < i)
    i, j = np.tril_indices(num_nodes, k=-1)

    # Reassign all existing weights (preserve structure but change values)
    mask = new_matrix[i, j] != 0
    new_weights = sample_from_disjoint_interval(mask.sum())
    new_matrix[i[mask], j[mask]] = new_weights

    # Get current removable and addable edges
    removable_edges = list(zip(i[mask], j[mask]))
    addable_edges = list(zip(i[~mask], j[~mask]))

    # Remove edges
    if mode in ['remove', 'both']:
        to_remove = random.sample(removable_edges, min(n, len(removable_edges)))
        rem_i, rem_j = zip(*to_remove) if to_remove else ([], [])
        new_matrix[rem_i, rem_j] = 0.0

    # Add edges
    if mode in ['add', 'both']:
        to_add = random.sample(addable_edges, min(n, len(addable_edges)))
        add_i, add_j = zip(*to_add) if to_add else ([], [])
        new_weights = sample_from_disjoint_interval(len(to_add)) if to_add else []
        new_matrix[add_i, add_j] = new_weights

    return new_matrix


def simulate_multigroup_data(p, s, c, n, PERT, shared_permutation=True, seed=None):
    rng = np.random.default_rng(seed)
    B = generate_connection_matrix(p, s)
    perm = None
    datasets = []
    matrices = []
    #sample_sizes = [rng.integers(int(10_000), int(15_000)) for _ in range(C)] #change 10_000 and 15_000 with 1.2*p and 2.5*p
    sample_size = n
    for g in range(c):
        n_perturb = int(np.count_nonzero(B) * PERT)
        Bc = perturb_adjacency_matrix(B, n_perturb, "both")
        variances = rng.uniform(1, 3, size=p)
        E = sample_uniform_noise(sample_size, p, variances)
        X, _, perm = generate_dataset(E, Bc, sample_size, permutation=perm 
        if shared_permutation else None)
        datasets.append(X)
        matrices.append(Bc)
    
    return datasets, matrices, perm



def average_squared_error(true_B, est_B):
    # Flatten matrices and compute MSE
    return mean_squared_error(true_B.flatten(), est_B.flatten())

def invert_permutation(B, perm):
    inverse_perm = np.argsort(perm)
    B_perm_inverted = B[inverse_perm, :][:, inverse_perm]
    return B_perm_inverted

def compute_rescaling_matrix(adj_matrix: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """
    Compute the rescaling matrix R such that:
    B = R * B_tilde

    Parameters:
    - adj_matrix: np.ndarray, shape (n, n)
        Adjacency matrix (non-zero entries represent edges i → j)
    - X: pd.DataFrame, shape (n_samples, n_variables)
        Original (non-normalized) data

    Returns:
    - R: np.ndarray, shape (n, n)
        Rescaling matrix
    """
    sums = X.sum().values
    n = adj_matrix.shape[0]
    R = np.zeros((n, n))
    for i in range(n):         # row: target node
        for j in range(n):     # col: source node
            if adj_matrix[i, j] != 0:
                R[i, j] = sums[i] / sums[j]
    return R

def normalize_by_column_sum(X):
    """
    Normalize each column so that the sum of each column is 1.
    
    Parameters:
    - X: np.ndarray, shape (n_rows, n_columns)
    
    Returns:
    - X_norm: np.ndarray, same shape as X
    """
    col_sum = X.sum(axis=0)
    col_sum[col_sum == 0] = 1  # Avoid division by zero
    X_norm = X / col_sum
    return X_norm



def run_experiment(p, s, n, r, PERT, c):
    X, Bs_true, perms = simulate_multigroup_data(p, s, c, n, PERT)
    X_normalized = [normalize_by_column_sum(X[0]), normalize_by_column_sum(X[1])]

    model = lingam.MultiGroupDirectLiNGAM()
    model.fit(X_normalized)

    mse_cumulative = 0
    shd_cumulative = 0
    precision_cumulative = 0
    recall_cumulative = 0

    for g in range(c):
        R = compute_rescaling_matrix(model.adjacency_matrices_[g], pd.DataFrame(X[g]))
        B_rescaled = model.adjacency_matrices_[g] * R
        B_est = invert_permutation(B_rescaled, perms)

        mse_ = average_squared_error(Bs_true[g], B_est)
        shd_ = SHD((Bs_true[g] != 0).astype(int), (B_est != 0).astype(int), True)

        # Precision and Recall
        B_true_bin = (Bs_true[g] != 0).astype(int)
        B_est_bin = (B_est != 0).astype(int)

        TP = ((B_true_bin == 1) & (B_est_bin == 1)).sum()
        FP = ((B_true_bin == 0) & (B_est_bin == 1)).sum()
        FN = ((B_true_bin == 1) & (B_est_bin == 0)).sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0



        mse_cumulative += mse_
        shd_cumulative += shd_
        precision_cumulative += precision
        recall_cumulative += recall

    return {
        'p': p,
        's': s,
        'n': n,
        'c': c,
        'perturbation': PERT,
        'run': r,
        'mse': mse_cumulative / c,
        'shd': shd_cumulative / c,
        'precision': precision_cumulative / c,
        'recall': recall_cumulative / c
    }
