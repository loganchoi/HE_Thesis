import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Generate dataset function (unchanged)
def generate_integer_dataset(m=400, n=20, seed=42):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = torch.randint(0, 10, (m, n), dtype=torch.float32)  # Random integers between 0 and 10
    Y = (X.sum(dim=1) % 2 == 0).float().unsqueeze(1)  # Binary label: 1 if sum is even, 0 if odd
    return X, Y

# Generate dataset
X, Y = generate_integer_dataset(seed=42)

# Define the logistic regression model (unchanged)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # Single output for binary classification

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Apply sigmoid to the linear output

# Initialize the logistic regression model
model = LogisticRegressionModel(X.shape[1])

# Add bias column of ones to X
ones = torch.ones(X.shape[0], 1, dtype=torch.float32)
X_with_bias = torch.cat([ones, X], dim=1)  # Shape (m, n+1)

# Compute the parameters using the normal equation: theta = (X^T X)^-1 X^T Y
X_transpose = X_with_bias.t()
theta = torch.inverse(X_transpose @ X_with_bias) @ X_transpose @ Y

# Display computed parameters (weights and bias)
# print(f"Model parameters (theta): {theta}")

# Predict using the computed model parameters
predicted = torch.sigmoid(X_with_bias @ theta)  # Apply sigmoid to the linear output

# Apply threshold to get predicted class (0 or 1)
predicted_classes = (predicted >= 0.5).float()

# Calculate accuracy
correct = (predicted_classes == Y).float().sum()  # Compare with original Y
accuracy = correct / len(Y)

print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")
print(classification_report(Y.numpy().squeeze(), predicted_classes.numpy().squeeze()))