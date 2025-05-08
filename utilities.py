import numpy as np

def generate_connection_matrix(p):
    s = 1 / (p - 1)  # Sparsity level

    # Generate a random matrix of same shape
    rand_matrix = np.random.binomial(1, s, size=(p, p))

    # Mask to keep only the strict lower triangular part (i > j)
    lower_triangle_mask = np.tril(np.ones((p, p)), k=-1)

    # Apply the mask
    matrix = rand_matrix * lower_triangle_mask

    return matrix
