import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load RNA-seq data
df = pd.read_csv(
    "/Users/loganchoi/Desktop/HE_Thesis/Data/new_filtered_prostate_cancer.csv",
    index_col=0
)

# Select 5 samples (rows) and 5 genes (columns)
df_subset = df.iloc[:5, :5]

# Create a heatmap (red color scheme)
plt.figure(figsize=(6, 5))
sns.heatmap(df_subset, cmap="Reds", annot=True, fmt=".2f", cbar_kws={'label': 'Expression Level'})

# Add labels and title at the bottom
plt.figtext(0.5, -0.1, "", ha="center", fontsize=14)
plt.xlabel("Genes")
plt.ylabel("Sample IDs")
plt.tight_layout()
plt.show()
