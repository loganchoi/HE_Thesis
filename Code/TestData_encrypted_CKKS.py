import torch
import tenseal as ts
import torch.nn as nn
import random
import time

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generate dataset function (unchanged)
def generate_float_dataset(m=400, n=20, seed=42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = torch.rand(m,n, dtype=torch.float32) * 10 # Random integers between 0 and 10
    Y = (X.round().sum(dim=1) % 2 == 0).float().unsqueeze(1)  # Binary label: 1 if sum is even, 0 if odd
    return X, Y

# Generate dataset
X, Y = generate_float_dataset(seed=42)

# TenSEAL setup for encryption
poly_mod_degree = 4096
plain_modulus = 786433
coeff_mod_bit_sizes = [40, 20, 40]

# Create TenSEAL context (for encrypted computations)
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, plain_modulus, coeff_mod_bit_sizes, ts.ENCRYPTION_TYPE.SYMMETRIC)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()

# Encrypt the dataset (X)
start = time.time()
enc_X = [ts.ckks_vector(context, x) for x in X]

# Encrypt the labels (Y)
enc_Y = [ts.ckks_vector(context, (y).int().tolist()) for y in Y]
end = time.time()
print(end-start)

# Define the logistic regression model (unchanged)
class EncryptedLogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(EncryptedLogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Single output for binary classification

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Apply sigmoid to the linear output

# Initialize the logistic regression model
model = EncryptedLogisticRegressionModel(X.shape[1])

# Decrypt data for direct computation (we're skipping training loop, so just decrypt for one pass)
decrypted_X = torch.tensor([enc_x.decrypt() for enc_x in enc_X], dtype=torch.float32)
decrypted_Y = torch.tensor([enc_y.decrypt() for enc_y in enc_Y], dtype=torch.float32)

# Add bias column of ones to X
ones = torch.ones(decrypted_X.shape[0], 1, dtype=torch.float32)
X_with_bias = torch.cat([ones, decrypted_X], dim=1)  # Shape (m, n+1)

# Compute the parameters using the normal equation: theta = (X^T X)^-1 X^T Y
X_transpose = X_with_bias.t()
theta = torch.inverse(X_transpose @ X_with_bias) @ X_transpose @ decrypted_Y

# Display computed parameters (weights and bias)
print(f"Model parameters (theta): {theta}")

# Predict using the computed model parameters
predicted = torch.sigmoid(X_with_bias @ theta)  # Apply sigmoid to the linear output

# Apply threshold to get predicted class (0 or 1)
predicted_classes = (predicted >= 0.5).float()

# Calculate accuracy
correct = (predicted_classes == decrypted_Y).float().sum()  # Compare with original Y (decrypted)
accuracy = correct / len(decrypted_Y)

print(f"Accuracy on training data: {accuracy}")
print(classification_report(Y.numpy().squeeze(), predicted_classes.numpy().squeeze()))