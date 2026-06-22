import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 5: Validate merged dataset and prepare final ML dataset
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

combined_file = DATA_DIR / "combined_fcbga_lid_dataset.csv"
df = pd.read_csv(combined_file)

# ------------------------------------------------------------
# Columns that should match between assembly and SJR datasets
# ------------------------------------------------------------
matching_columns = [
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

print("\nChecking if assembly and SJR design variables match:\n")

all_match = True

for col in matching_columns:
    assembly_col = col + "_assembly"
    sjr_col = col + "_sjr"

    if assembly_col in df.columns and sjr_col in df.columns:
        match = (df[assembly_col].astype(str) == df[sjr_col].astype(str)).all()
        print(f"{col}: {'MATCH' if match else 'NOT MATCH'}")

        if not match:
            all_match = False
            mismatch_count = (df[assembly_col].astype(str) != df[sjr_col].astype(str)).sum()
            print(f"  Mismatch count: {mismatch_count}")
    else:
        print(f"{col}: column missing")
        all_match = False

# ------------------------------------------------------------
# Prepare final ML dataset
# Use only one copy of common design variables
# ------------------------------------------------------------
final_columns = [
    "Design_ID",

    # Common design inputs
    "Cu-pillar pitch–diameter_assembly",
    "Bulk silicon thickness_assembly",
    "Bump solder height_assembly",
    "Substrate core thickness_assembly",
    "Substrate core (E, CTE)_assembly",
    "UF (E, CTE)_assembly",
    "Lid foot width_assembly",
    "Lid thickness_assembly",
    "Cu-pillar bump solder material_assembly",

    # Extra SJR input
    "BGA solder material",

    # Targets
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach",
    "DeltaW_BGA",
    "DeltaW_bump"
]

final_df = df[final_columns].copy()

# Rename columns to clean names
final_df = final_df.rename(columns={
    "Cu-pillar pitch–diameter_assembly": "Cu-pillar pitch–diameter",
    "Bulk silicon thickness_assembly": "Bulk silicon thickness",
    "Bump solder height_assembly": "Bump solder height",
    "Substrate core thickness_assembly": "Substrate core thickness",
    "Substrate core (E, CTE)_assembly": "Substrate core (E, CTE)",
    "UF (E, CTE)_assembly": "UF (E, CTE)",
    "Lid foot width_assembly": "Lid foot width",
    "Lid thickness_assembly": "Lid thickness",
    "Cu-pillar bump solder material_assembly": "Cu-pillar bump solder material"
})

print("\nFinal ML dataset shape:")
print(final_df.shape)

print("\nFinal ML dataset columns:")
for col in final_df.columns:
    print("-", col)

print("\nMissing values in final ML dataset:")
print(final_df.isnull().sum())

# Save final dataset
final_file = DATA_DIR / "final_fcbga_lid_ml_dataset.csv"
final_df.to_csv(final_file, index=False)

print(f"\nFinal ML dataset saved to: {final_file}")

if all_match:
    print("\nValidation result: All common design variables match.")
else:
    print("\nValidation result: Some common design variables do NOT match. Please review before ML.")

print("\nStep 5 completed successfully.")