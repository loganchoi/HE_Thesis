import pandas as pd

# Load the dataset
# Replace 'framingham.csv' with the path to your dataset
framingham_data = pd.read_csv('framingham.csv')

# Display data types for all columns
print(framingham_data.dtypes)

framingham_data.info()

diabetes_data = pd.read_csv('diabetes2.csv')

# Display data types for all columns
print(diabetes_data.dtypes)

diabetes_data.info()

cancer_data = pd.read_csv('Breast_cancer_data.csv')

# Display data types for all columns
print(cancer_data.dtypes)

cancer_data.info()




