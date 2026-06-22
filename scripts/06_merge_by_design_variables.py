import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 6: Correct merge using design variables, not row number
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

assembly_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
sjr_file = DATA_DIR / "cleaned_sjr_lid_data.csv"

assembly_df = pd.read_csv(assembly_file)
sjr_df = pd.read_csv(sjr_file)

# Clean column names
assembly_df.columns = assembly_df.columns.str.strip()
sjr_df.columns = sjr_df.columns.str.strip()

# ------------------------------------------------------------
# Rename columns so both datasets use the same names
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
# Common design variables used for correct merging
# ------------------------------------------------------------
merge_columns = [
    "Cu-pillar pitch–diameter",
    "Bulk silicon thickness",
    "Bump solder height",
    "Substrate core thickness",
    "Substrate core (E, CTE)",
    "UF (E, CTE)",
    "Lid foot width",
    "Lid thickness",
    "Cu-pillar bump solder material"
]

assembly_targets = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach"
]

sjr_targets = [
    "BGA solder material",
    "DeltaW_BGA",
    "DeltaW_bump"
]

# ------------------------------------------------------------
# Check duplicated design points before merge
# ------------------------------------------------------------
print("\nAssembly shape:")
print(assembly_df.shape)

print("\nSJR shape:")
print(sjr_df.shape)

assembly_duplicates = assembly_df.duplicated(subset=merge_columns).sum()
sjr_duplicates = sjr_df.duplicated(subset=merge_columns).sum()

print("\nDuplicate design points based on merge columns:")
print("Assembly duplicates:", assembly_duplicates)
print("SJR duplicates:", sjr_duplicates)

# ------------------------------------------------------------
# Merge based on actual design variables
# ------------------------------------------------------------
combined_df = pd.merge(
    assembly_df[merge_columns + assembly_targets],
    sjr_df[merge_columns + sjr_targets],
    on=merge_columns,
    how="inner"
)

# Add Design_ID after correct merge
combined_df.insert(0, "Design_ID", range(1, len(combined_df) + 1))

print("\nCorrectly merged dataset shape:")
print(combined_df.shape)

print("\nCorrectly merged columns:")
for col in combined_df.columns:
    print("-", col)

print("\nMissing values:")
print(combined_df.isnull().sum())

# ------------------------------------------------------------
# Save corrected merged dataset
# ------------------------------------------------------------
output_file = DATA_DIR / "final_fcbga_lid_ml_dataset_correct_merge.csv"
combined_df.to_csv(output_file, index=False)

print(f"\nCorrect merged dataset saved to: {output_file}")

# ------------------------------------------------------------
# Important check
# ------------------------------------------------------------
if len(combined_df) == 300:
    print("\nMerge result: Perfect. All 300 design points matched.")
elif len(combined_df) == 0:
    print("\nMerge result: No rows matched. The two datasets may use different design spaces.")
else:
    print(f"\nMerge result: Only {len(combined_df)} rows matched. Review before ML.")

print("\nStep 6 completed successfully.")