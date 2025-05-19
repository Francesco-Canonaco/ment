
import numpy as np

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

def generate_dataset(E, B, permutation=None):
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



