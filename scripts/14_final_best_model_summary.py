import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 14: Final best model summary after tuning
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

# Load previous model results
rf_xgb_file = RESULTS_DIR / "rf_vs_xgboost_all_results.csv"
tuned_file = RESULTS_DIR / "tuned_weak_model_results.csv"

rf_xgb_df = pd.read_csv(rf_xgb_file)
tuned_df = pd.read_csv(tuned_file)

# Rename tuned model column to match Model column
tuned_df = tuned_df.rename(columns={
    "Best_Tuned_Model": "Model"
})

# Keep same columns
tuned_df = tuned_df[
    [
        "Dataset",
        "Target",
        "Model",
        "Train_R2",
        "Test_R2",
        "Train_MSE",
        "Test_MSE"
    ]
]

# Combine all model results
all_models_df = pd.concat([rf_xgb_df, tuned_df], ignore_index=True)

# Select best model for each target based on Test_R2
best_models_df = all_models_df.loc[
    all_models_df.groupby(["Dataset", "Target"])["Test_R2"].idxmax()
].reset_index(drop=True)

# Add model quality label
def quality_label(r2):
    if r2 >= 0.90:
        return "Very good"
    elif r2 >= 0.75:
        return "Good"
    elif r2 >= 0.50:
        return "Medium"
    else:
        return "Weak"

best_models_df["Model_Quality"] = best_models_df["Test_R2"].apply(quality_label)

# Sort for clean display
dataset_order = {"Assembly": 0, "SJR": 1}
best_models_df["Dataset_Order"] = best_models_df["Dataset"].map(dataset_order)
best_models_df = best_models_df.sort_values(
    by=["Dataset_Order", "Target"]
).drop(columns=["Dataset_Order"])

print("\nFinal best model summary after tuning:")
print(best_models_df)

# Save final files
all_models_output = RESULTS_DIR / "all_models_including_tuned_results.csv"
best_models_output = RESULTS_DIR / "final_best_model_summary_after_tuning.csv"

all_models_df.to_csv(all_models_output, index=False)
best_models_df.to_csv(best_models_output, index=False)

print(f"\nAll models saved to: {all_models_output}")
print(f"Final best model summary saved to: {best_models_output}")

print("\nStep 14 completed successfully.")