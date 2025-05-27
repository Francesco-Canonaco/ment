from utilities import run_experiment
from itertools import product
import pandas as pd
from joblib import Parallel, delayed

P = [50, 100, 150, 200] # number of variables/features
S = [0.2, 0.25, 0.3] # edge probabilities (control the sparsity)
N = [10_000, 20_000, 30_000] # number of samples 
PERT = 0.10 # perturbation level (number of perturbation is |E|*PERT)
RUNS = 100 # number of runs for each triple: <P,S,N>
c = 2 # number of groups

# Run in parallel
results_list = Parallel(n_jobs=-1)(  # set n_jobs=N to limit cores
    delayed(run_experiment)(p, s, n, r, PERT, c)
    for p, s, n in product(P, S, N)
    for r in range(RUNS)
)

# Convert to DataFrame
results_df = pd.DataFrame(results_list)
results_df.to_csv("results.csv")
