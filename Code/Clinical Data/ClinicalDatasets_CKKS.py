import torch
import tenseal as ts
import pandas as pd
import random
from time import time
import numpy as np
# optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def split_train_test(x, y, test_ratio=0.3,seed =42):
    idxs = [i for i in range(len(x))]
    rng = random.Random(seed)
    rng.shuffle(idxs)
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def heart_disease_data():
    print("Heart Disease Data")
    data = pd.read_csv("../../Data/framingham.csv")
    data = data.dropna()
    data = data.drop(columns=[])
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop("TenYearCHD", axis="columns")
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    column_names = data.columns.to_list()
    return split_train_test(x, y) + (column_names,)

def breast_cancer_data():
    print("Breast Cancer Data")
    data = pd.read_csv("../../Data/Breast_cancer_data.csv")
    data = data.dropna()
    data = data.drop(columns=[])
    grouped = data.groupby("diagnosis")
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    y = torch.tensor(data["diagnosis"].values).float().unsqueeze(1)
    data = data.drop("diagnosis", axis="columns")
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    column_names = data.columns.to_list()
    return split_train_test(x, y) + (column_names,)

def diabetes_data():
    print("Diabetes Data")
    data = pd.read_csv("../../Data/diabetes2.csv")
    data = data.dropna()
    data = data.drop(columns=[])
    grouped = data.groupby("Outcome")
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))
    y = torch.tensor(data["Outcome"].values).float().unsqueeze(1)
    data = data.drop("Outcome", axis="columns")
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    column_names = data.columns.to_list()
    return split_train_test(x, y) + (column_names,)


x_train, y_train, x_test, y_test, column_names = diabetes_data()

print("CKKS\n")
print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################\n")

#############################
# Plain Logistic Regression
class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out
    

n_features = x_train.shape[1]
model = LR(n_features)
optim = torch.optim.SGD(model.parameters(), lr=1)
criterion = torch.nn.BCELoss()
EPOCHS = 5

def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        #print(f"Loss at epoch {e}: {loss.data}")
    return model

model = train(model, optim, criterion, x_train, y_train)
print("\n")

def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()

plain_accuracy = accuracy(model, x_test, y_test)
print(f"Accuracy on plain test_set: {plain_accuracy}")

#############################
# Encrypted Logistic Regression
class EncryptedLR:
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

eelr = EncryptedLR(model)

# TenSEAL context
poly_mod_degree = 8192
plain_modulus = 786433
coeff_mod_bit_sizes = [30,20,30]
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree,plain_modulus, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 22
ctx_eval.generate_galois_keys()

# Encrypt test set
t_start = time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
t_end = time()
print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")

def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time()
    
    correct = 0
    y_pred_encrypted = []
    for enc_x, y in zip(enc_x_test, y_test):
        enc_out = model(enc_x)
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)
        pred = (out >= 0.5).float()
        y_pred_encrypted.append(pred.item())
        if torch.abs(out - y) < 0.5:
            correct += 1
    
    t_end = time()
    print(f"Evaluated test_set of {len(x_test)} entries in {int(t_end - t_start)} seconds")
    accuracy = correct / len(x_test)
    print(f"Accuracy: {correct}/{len(x_test)} = {accuracy}")
    return accuracy, y_pred_encrypted

encrypted_accuracy, y_pred_encrypted = encrypted_evaluation(eelr, enc_x_test, y_test)
diff_accuracy = plain_accuracy - encrypted_accuracy
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")

print()
print("Feature Importance CKKS:")
sorted_features = sorted(zip(column_names, eelr.weight), key=lambda x: abs(x[1]), reverse=True)
for name, weight in sorted_features:
    print(f"{name}: {weight}")

print()
print("Feature Importance Plain:")
sorted_features = sorted(zip(column_names, model.lr.weight.data[0]), key=lambda x: abs(x[1]), reverse=True)
for name, weight in sorted_features:
    print(f"{name}: {weight}")
#############################
# Classification Reports

# Plain model predictions
y_pred_plain_probs = model(x_test)
y_pred_plain = (y_pred_plain_probs >= 0.5).float()

print("\nClassification Report (Plain Model):")
print(classification_report(y_test.numpy(), y_pred_plain.numpy()))

print("Classification Report (Encrypted Model):")
print(classification_report(y_test.numpy(), y_pred_encrypted))
