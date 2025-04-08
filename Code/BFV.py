import tenseal as ts
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1️⃣ Load and Preprocess Data (Ensure Integer Scaling)
data = np.random.randn(1000, 10)  # Simulated dataset
labels = np.random.randint(0, 2, 1000)

scaler = StandardScaler()
data = scaler.fit_transform(data)

scale_factor = 1000  # Scale float values to integers
data = (data * scale_factor).astype(int)  # Convert to integers

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 2️⃣ Setup BFV Encryption Context
poly_mod_degree = 8192  # Larger value improves security
plain_modulus = 1032193  # Large prime number for integer encoding

ctx_eval = ts.context(
    ts.SCHEME_TYPE.BFV,  # Use BFV scheme
    poly_mod_degree,
    plain_modulus,
    [], # Required for BFV
)
ctx_eval.generate_galois_keys()
ctx_eval.generate_relin_keys()

# 3️⃣ Encrypt Data with BFV
enc_x_test = [ts.bfv_vector(ctx_eval, x.tolist()) for x in x_test]

# 4️⃣ Define Encrypted Logistic Regression Class
class EncryptedLR:
    def __init__(self, ctx, input_size):
        self.ctx = ctx
        self.weight = ts.bfv_vector(ctx, np.random.randint(-10, 10, input_size).tolist())
        self.bias = ts.bfv_vector(ctx, [0])

    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias  # Linear function
        enc_out = self.approx_sigmoid(enc_out)  # Apply sigmoid approximation
        return enc_out
    
    @staticmethod
    def approx_sigmoid(enc_x):
        """Polynomial approximation of sigmoid."""
        return 0.5 + 0.125 * enc_x - 0.0025 * (enc_x * enc_x)  # Quadratic approximation

# 5️⃣ Perform Encrypted Inference
model = EncryptedLR(ctx_eval, input_size=10)
predictions = [model.forward(enc_x) for enc_x in enc_x_test]

print("Encrypted inference complete!")
