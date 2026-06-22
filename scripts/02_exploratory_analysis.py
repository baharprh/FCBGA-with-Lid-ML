import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 2: Exploratory Data Analysis
# ============================================================

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
data_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
df = pd.read_csv(data_file)

# Clean column names
df.columns = df.columns.str.strip()

# Target columns
target_columns = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid atach"
]

input_columns = [col for col in df.columns if col not in target_columns]

# ------------------------------------------------------------
# 1. Print basic information
# ------------------------------------------------------------
print("\nDataset shape:")
print(df.shape)

print("\nInput columns:")
for col in input_columns:
    print("-", col)

print("\nTarget columns:")
for col in target_columns:
    print("-", col)

# ------------------------------------------------------------
# 2. Identify numerical and categorical columns
# ------------------------------------------------------------
categorical_columns = df[input_columns].select_dtypes(include=["object"]).columns.tolist()
numerical_columns = df[input_columns].select_dtypes(exclude=["object"]).columns.tolist()

print("\nNumerical input columns:")
for col in numerical_columns:
    print("-", col)

print("\nCategorical input columns:")
for col in categorical_columns:
    print("-", col)

# ------------------------------------------------------------
# 3. Show unique values for categorical columns
# ------------------------------------------------------------
print("\nUnique values in categorical columns:")
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].unique())

# ------------------------------------------------------------
# 4. Descriptive statistics
# ------------------------------------------------------------
print("\nDescriptive statistics for all numerical columns:")
print(df.describe())

# Save descriptive statistics
stats_file = RESULTS_DIR / "descriptive_statistics.csv"
df.describe().to_csv(stats_file)

print(f"\nDescriptive statistics saved to: {stats_file}")

# ------------------------------------------------------------
# 5. Plot target distributions
# ------------------------------------------------------------
for target in target_columns:
    plt.figure(figsize=(7, 5))
    plt.hist(df[target], bins=20, edgecolor="black")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {target}")
    plt.grid(True, linestyle="--", alpha=0.5)

    figure_path = FIGURES_DIR / f"distribution_{target.replace(' ', '_')}.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")

# ------------------------------------------------------------
# 6. Correlation heatmap for numerical columns
# ------------------------------------------------------------
numeric_df = df.select_dtypes(exclude=["object"])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="viridis", aspect="auto")
plt.colorbar(label="Correlation coefficient")

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()

heatmap_path = FIGURES_DIR / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure: {heatmap_path}")

print("\nStep 2 completed successfully.")