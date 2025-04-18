import os
import time
import torch
import random
import numpy as np
import tenseal as ts
from sklearn.metrics import classification_report

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generate dataset function (unchanged)
def generate_integer_dataset(m,n,seed=42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = torch.randint(0, 10, (m, n), dtype=torch.float32)  # Random integers between 0 and 10
    Y = (X.sum(dim=1) % 2 == 0).float().unsqueeze(1)  # Binary label: 1 if sum is even, 0 if odd
    return X, Y

# Generate dataset
m = 1000
n = 10
X, Y = generate_integer_dataset(m,n,seed=42)

# TenSEAL setup for encryption
poly_mod_degree = 8192
plain_modulus = 786433
coeff_mod_bit_sizes = [30, 20, 30]

# Create TenSEAL context (for encrypted computations)
context = ts.context(ts.SCHEME_TYPE.BFV, poly_mod_degree, plain_modulus, coeff_mod_bit_sizes, ts.ENCRYPTION_TYPE.SYMMETRIC)
context.generate_galois_keys()
context.generate_relin_keys()

# Start encryption
start = time.time()
enc_X = [ts.bfv_vector(context, x) for x in X]
enc_Y = [ts.bfv_vector(context, y) for y in Y]
end = time.time()

# Print the results
print("BFV")
print("POLY_MOD_DEGREE", poly_mod_degree)
print("Data Rows:", m)
print(f"Encryption took: {end - start:.4f} seconds")
