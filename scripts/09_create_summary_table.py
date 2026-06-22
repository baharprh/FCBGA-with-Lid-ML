import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 9: Create summary table for all ML results
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

assembly_file = RESULTS_DIR / "assembly_model_performance.csv"
sjr_file = RESULTS_DIR / "sjr_model_performance.csv"

assembly_df = pd.read_csv(assembly_file)
sjr_df = pd.read_csv(sjr_file)

assembly_df.insert(0, "Dataset", "Assembly")
sjr_df.insert(0, "Dataset", "SJR")

summary_df = pd.concat([assembly_df, sjr_df], ignore_index=True)

# Add simple quality label based on Test R2
def quality_label(r2):
    if r2 >= 0.90:
        return "Very good"
    elif r2 >= 0.75:
        return "Good"
    elif r2 >= 0.50:
        return "Medium"
    else:
        return "Weak"

summary_df["Model_Quality"] = summary_df["Test_R2"].apply(quality_label)

# Reorder columns
summary_df = summary_df[
    [
        "Dataset",
        "Target",
        "Train_R2",
        "Test_R2",
        "Train_MSE",
        "Test_MSE",
        "Model_Quality"
    ]
]

print("\nFinal ML performance summary:")
print(summary_df)

output_file = RESULTS_DIR / "final_model_performance_summary.csv"
summary_df.to_csv(output_file, index=False)

print(f"\nFinal summary saved to: {output_file}")
print("\nStep 9 completed successfully.")