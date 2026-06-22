import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 1: Check and clean the dataset
# ============================================================

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Excel file path
excel_file = DATA_DIR / "2D_assembly_lid_300_v4.xlsx"

# ------------------------------------------------------------
# 1. Check if the Excel file exists
# ------------------------------------------------------------
if not excel_file.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_file}")

# ------------------------------------------------------------
# 2. Print sheet names
# ------------------------------------------------------------
excel_data = pd.ExcelFile(excel_file)

print("\nSheet names:")
for sheet in excel_data.sheet_names:
    print("-", sheet)

# ------------------------------------------------------------
# 3. Load the main sheet
# ------------------------------------------------------------
sheet_name = "2D_assembly_lid_300"

df = pd.read_excel(excel_file, sheet_name=sheet_name)

# ------------------------------------------------------------
# 4. Clean column names
# This removes extra spaces before/after column names
# ------------------------------------------------------------
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# 5. Basic dataset information
# ------------------------------------------------------------
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for col in df.columns:
    print("-", col)

# ------------------------------------------------------------
# 6. Check missing values
# ------------------------------------------------------------
print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 7. Define target/output columns
# These are the values we want the ML model to predict
# ------------------------------------------------------------
target_columns = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid atach"
]

# ------------------------------------------------------------
# 8. Check if target columns exist
# ------------------------------------------------------------
print("\nChecking target columns:")
for target in target_columns:
    if target in df.columns:
        print(f"FOUND: {target}")
    else:
        print(f"NOT FOUND: {target}")

# Stop the code if a target column is missing
missing_targets = [target for target in target_columns if target not in df.columns]

if missing_targets:
    raise ValueError(f"These target columns were not found: {missing_targets}")

# ------------------------------------------------------------
# 9. Define input columns
# These are the design parameters used to predict the targets
# ------------------------------------------------------------
input_columns = [col for col in df.columns if col not in target_columns]

print("\nInput columns:")
for col in input_columns:
    print("-", col)

print("\nTarget columns:")
for col in target_columns:
    print("-", col)

print("\nNumber of input columns:", len(input_columns))
print("Number of target columns:", len(target_columns))

# ------------------------------------------------------------
# 10. Save cleaned full dataset as CSV
# ------------------------------------------------------------
cleaned_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
df.to_csv(cleaned_file, index=False)

print(f"\nCleaned CSV saved to: {cleaned_file}")

# ------------------------------------------------------------
# 11. Save input and target datasets separately
# ------------------------------------------------------------
X = df[input_columns]
y = df[target_columns]

X_file = DATA_DIR / "input_features.csv"
y_file = DATA_DIR / "target_outputs.csv"

X.to_csv(X_file, index=False)
y.to_csv(y_file, index=False)

print(f"Input features saved to: {X_file}")
print(f"Target outputs saved to: {y_file}")

print("\nStep 1 completed successfully.")