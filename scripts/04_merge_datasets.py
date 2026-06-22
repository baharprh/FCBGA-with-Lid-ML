import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 4: Merge Assembly and SJR datasets
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load cleaned datasets
assembly_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
sjr_file = DATA_DIR / "cleaned_sjr_lid_data.csv"

assembly_df = pd.read_csv(assembly_file)
sjr_df = pd.read_csv(sjr_file)

# Clean column names
assembly_df.columns = assembly_df.columns.str.strip()
sjr_df.columns = sjr_df.columns.str.strip()

# ------------------------------------------------------------
# Rename columns to make both datasets consistent
# ------------------------------------------------------------
assembly_df = assembly_df.rename(columns={
    "Lid thicknes": "Lid thickness",
    "Bump solder material": "Cu-pillar bump solder material",
    "Warpage post lid atach": "Warpage post lid attach"
})

sjr_df = sjr_df.rename(columns={
    "lid thickness": "Lid thickness"
})

# ------------------------------------------------------------
# Add row ID because both datasets have 300 design points
# We assume row 1 in assembly corresponds to row 1 in SJR
# ------------------------------------------------------------
assembly_df["Design_ID"] = range(1, len(assembly_df) + 1)
sjr_df["Design_ID"] = range(1, len(sjr_df) + 1)

# ------------------------------------------------------------
# Merge by Design_ID
# ------------------------------------------------------------
combined_df = pd.merge(
    assembly_df,
    sjr_df,
    on="Design_ID",
    how="inner",
    suffixes=("_assembly", "_sjr")
)

# ------------------------------------------------------------
# Print information
# ------------------------------------------------------------
print("\nAssembly shape:")
print(assembly_df.shape)

print("\nSJR shape:")
print(sjr_df.shape)

print("\nCombined shape:")
print(combined_df.shape)

print("\nCombined columns:")
for col in combined_df.columns:
    print("-", col)

print("\nMissing values in combined dataset:")
print(combined_df.isnull().sum())

# ------------------------------------------------------------
# Save combined dataset
# ------------------------------------------------------------
combined_file = DATA_DIR / "combined_fcbga_lid_dataset.csv"
combined_df.to_csv(combined_file, index=False)

print(f"\nCombined dataset saved to: {combined_file}")
print("\nStep 4 completed successfully.")