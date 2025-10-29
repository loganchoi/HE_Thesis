import csv
import os
import pandas as pd

input_file = "/Users/loganchoi/Desktop/HE_Thesis/Data/TcgaTargetGtex_rsem_gene_fpkm_converted.csv"

def breast_cancer_data():
    print("BREAST CANCER")
    output_file = "/Users/loganchoi/Desktop/HE_Thesis/Data/whole_dataset_breast_cancer.csv"
    return "/Users/loganchoi/Desktop/HE_Thesis/Data/Breast_cancer_genomic.csv", output_file

def lung_cancer_data():
    print("LUNG CANCER")
    output_file = "/Users/loganchoi/Desktop/HE_Thesis/Data/whole_dataset_lung_cancer.csv"
    return "/Users/loganchoi/Desktop/HE_Thesis/Data/lung_cancer_genomic.csv", output_file

def prostate_cancer_data():
    print("PROSTATE CANCER")
    output_file = "/Users/loganchoi/Desktop/HE_Thesis/Data/whole_dataset_prostate_cancer.csv"
    return "/Users/loganchoi/Desktop/HE_Thesis/Data/Prostate_cancer_genomic.csv", output_file

keepFile, output_file = prostate_cancer_data()
data = pd.read_csv(keepFile)
keep = set(data["sample"])

breast_ENSGs = {
    "ENSG00000012048.19",
    "ENSG00000139618.14",
    "ENSG00000196569.11",
    "ENSG00000157150.4",
    "ENSG00000133687.15",
    "ENSG00000091831.21", 
    "ENSG00000140009.18",
    "ENSG00000060718.18",
    "ENSG00000123500.9",
    "ENSG00000146374.13"
}

lung_ENSGs = {
    "ENSG00000141510.15",
    "ENSG00000179603.17",
    "ENSG00000142208.15",
    "ENSG00000178568.13",
    "ENSG00000159216.18",
    "ENSG00000079999.13",
    "ENSG00000109670.13",
    "ENSG00000133703.11",
    "ENSG00000181449.3",
    "ENSG00000077782.19"
}

prostate_ENSGs = {
    "ENSG00000133740.10",
    "ENSG00000157554.18",
    "ENSG00000171862.9",
    "ENSG00000129514.5", 
    "ENSG00000136997.14",
    "ENSG00000169083.15",
    "ENSG00000139687.13",
    "ENSG00000134954.14",
    "ENSG00000157557.11",
    "ENSG00000164683.16"
}
# Customize your filter logic for column header
def keep_column(header_name):
    return header_name in keep  # <-- Change as needed

# Filter function to check if a certain row should be kept (e.g., checking a specific value in a column)
def filter_row(row,row_count):
    # Example: Check if the value in the first column (gene name) matches 'ENSG'
    if row_count < 10000:
        return True
    return False # Adjust the condition based on what you want to filter

print("Reading and filtering...")

with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)

    # Figure out which columns to keep (excluding 0, which is gene name)
    keep_indices = [i for i, name in enumerate(header) if keep_column(name)]
    keep_indices = [0] + keep_indices  # Include gene name column

    # Prepare a list of lists for the transposed data (empty rows)
    transposed = [[] for _ in range(len(keep_indices))]

    # Add header values
    for i, idx in enumerate(keep_indices):
        transposed[i].append(header[idx])

    # Process the file row-by-row
    row_count = 0
    for row in reader:
        if filter_row(row,row_count):  # Only include rows that match the filter condition
            for i, idx in enumerate(keep_indices):
                transposed[i].append(row[idx])
        row_count += 1
        if row_count % 10000 == 0:
            print(f"Processed {row_count} rows...")

print("Writing filtered data...")

# Write the filtered data to the output file
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(transposed)

print(f"Done. Filtered file written to: {output_file}")
