import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 3: Check SJR dataset
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

excel_file = DATA_DIR / "2D_SJR_lid_300_v4.xlsx"

if not excel_file.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_file}")

# Read Excel file
excel_data = pd.ExcelFile(excel_file)

print("\nSheet names:")
for sheet in excel_data.sheet_names:
    print("-", sheet)

# Load first sheet automatically
sheet_name = excel_data.sheet_names[0]
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Clean column names
df.columns = df.columns.str.strip()

print("\nLoaded sheet:")
print(sheet_name)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
for col in df.columns:
    print("-", col)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())

# Save cleaned SJR data
cleaned_file = DATA_DIR / "cleaned_sjr_lid_data.csv"
df.to_csv(cleaned_file, index=False)

print(f"\nCleaned SJR CSV saved to: {cleaned_file}")
print("\nStep 3 completed successfully.")