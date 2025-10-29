import torch
import tenseal as ts
import pandas as pd
import random
from time import time

# optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#merged_df = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))

def split_train_test(x, y, test_ratio=0.3,seed =42):
    idxs = [i for i in range(len(x))]
    rng = random.Random(seed)
    rng.shuffle(idxs)
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def prostate_cancer_data():
    print("Prostate Cancer")
    df1 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/new_filtered_prostate_cancer.csv")
    df2 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/Prostate_cancer_genomic.csv")

    df2 = df2[["sample","_sample_type"]]
    merged_df = pd.merge(df1, df2, on="sample", how="inner")

    merged_df["_sample_type"] = merged_df["_sample_type"].apply(
        lambda s: 0 if "normal" in str(s).lower() else 1
        )
    merged_df = merged_df.drop(columns=["sample"])
    merged_df = merged_df.sample(frac=1, random_state=73).reset_index(drop=True)

    y = torch.tensor(merged_df["_sample_type"].values).float().unsqueeze(1)
    merged_df = merged_df.drop("_sample_type", axis="columns")
    merged_df = (merged_df - merged_df.mean()) / merged_df.std()
    x = torch.tensor(merged_df.values).float()
    column_names = merged_df.columns.to_list()
    return split_train_test(x, y) + (column_names,)

def breast_cancer_data():
    print("Breast Cancer")
    df1 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/new_filtered_breast_cancer.csv")
    df2 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/Breast_cancer_genomic.csv")

    df2 = df2[["sample","_sample_type"]]
    merged_df = pd.merge(df1, df2, on="sample", how="inner")

    merged_df["_sample_type"] = merged_df["_sample_type"].apply(
        lambda s: 0 if "normal" in str(s).lower() else 1
        )
    merged_df = merged_df.drop(columns=["sample"])
    merged_df = merged_df.sample(frac=1, random_state=73).reset_index(drop=True)

    y = torch.tensor(merged_df["_sample_type"].values).float().unsqueeze(1)
    merged_df = merged_df.drop("_sample_type", axis="columns")
    merged_df = (merged_df - merged_df.mean()) / merged_df.std()
    x = torch.tensor(merged_df.values).float()
    column_names = merged_df.columns.to_list()
    return split_train_test(x, y) + (column_names,)

def lung_cancer_data():
    print("Lung Cancer")
    df1 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/new_filtered_lung_cancer.csv")
    df2 = pd.read_csv("/Users/loganchoi/Desktop/HE_Thesis/Data/lung_cancer_genomic.csv")

    df2 = df2[["sample","_sample_type"]]
    merged_df = pd.merge(df1, df2, on="sample", how="inner")

    merged_df["_sample_type"] = merged_df["_sample_type"].apply(
        lambda s: 0 if "normal" in str(s).lower() else 1
        )
    merged_df = merged_df.drop(columns=["sample"])
    merged_df = merged_df.sample(frac=1, random_state=73).reset_index(drop=True)

    y = torch.tensor(merged_df["_sample_type"].values).float().unsqueeze(1)
    merged_df = merged_df.drop("_sample_type", axis="columns")
    merged_df = (merged_df - merged_df.mean()) / merged_df.std()
    x = torch.tensor(merged_df.values).float()
    column_names = merged_df.columns.to_list()
    return split_train_test(x, y) + (column_names,)


x_train, y_train, x_test, y_test,column_names = breast_cancer_data()

print("BFV\n")
print("############# Data summary #############")
print(f"x_train has shape: {x_train.shape}")
print(f"y_train has shape: {y_train.shape}")
print(f"x_test has shape: {x_test.shape}")
print(f"y_test has shape: {y_test.shape}")
print("#######################################\n")

class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.lr(x))

n_features = x_train.shape[1]
model = LR(n_features)
optim = torch.optim.SGD(model.parameters(), lr=1)
criterion = torch.nn.BCELoss()
EPOCHS = 5

def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(epochs):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model


model = train(model, optim, criterion, x_train, y_train)

print("\n")

def accuracy(model, x, y):
    preds = model(x)
    correct = torch.abs(preds - y) < 0.5
    return correct.float().mean()

plain_acc = accuracy(model, x_test, y_test)
print(f"Plain test accuracy: {plain_acc}")

#####################
### BFV Version #####
#####################

# # BFV requires integer inputs, so scale and convert
SCALING_FACTOR = 100000
x_test_int = (x_test * SCALING_FACTOR).int()

class EncryptedLR_BFV:
    def __init__(self, model, scale):
        self.weight = [int(w * scale) for w in model.lr.weight.data[0]]
        self.bias = int(model.lr.bias.item() * scale)
        self.scale = scale

    def forward(self, enc_x):
        dot = enc_x.dot(self.weight)
        return dot + self.bias

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# Setup BFV context
bfv_poly_mod_degree = 8192
bfv_plain_modulus = 786433
coeff_mod_bit_sizes = [30,20,30]
ctx_bfv = ts.context(ts.SCHEME_TYPE.BFV, bfv_poly_mod_degree, bfv_plain_modulus,coeff_mod_bit_sizes)
ctx_bfv.generate_galois_keys()

eelr_bfv = EncryptedLR_BFV(model, SCALING_FACTOR)

# Encrypt test set
t_enc_start = time()
enc_x_test_bfv = [ts.bfv_vector(ctx_bfv, x.tolist()) for x in x_test]
t_enc_end = time()
print(f"Encryption of the test-set took {int(t_enc_end - t_enc_start)} seconds")

def bfv_encrypted_eval(model, enc_x_test, y_test):
    correct = 0
    t_start = time()
    y_pred_encrypted = []

    for enc_x, y in zip(enc_x_test, y_test):
        enc_out = model(enc_x)
        out = enc_out.decrypt()[0] / SCALING_FACTOR
        out = 1 / (1 + torch.exp(-torch.tensor(out)))  # Sigmoid on decrypted
        pred = (out >= 0.5).float()
        y_pred_encrypted.append(pred.item())
        if torch.abs(out - y) < 0.5:
            correct += 1

    t_end = time()
    acc = correct / len(enc_x_test)
    print(f"Encrypted test-set evaluation took {int(t_end - t_start)} seconds")
    print(f"Encrypted Accuracy: {acc:.4f}")
    return acc, y_pred_encrypted

bfv_acc, y_pred_encrypted = bfv_encrypted_eval(eelr_bfv, enc_x_test_bfv, y_test)
diff_accuracy = plain_acc - bfv_acc
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")

print()
print("Feature Importance BFV:")
sorted_features = sorted(zip(column_names, eelr_bfv.weight), key=lambda x: abs(x[1]), reverse=True)
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
