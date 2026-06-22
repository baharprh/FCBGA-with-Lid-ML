import pandas as pd
from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 12: Compare Random Forest and XGBoost results
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

# Load Random Forest results
assembly_rf = pd.read_csv(RESULTS_DIR / "assembly_model_performance.csv")
sjr_rf = pd.read_csv(RESULTS_DIR / "sjr_model_performance.csv")

assembly_rf.insert(0, "Dataset", "Assembly")
sjr_rf.insert(0, "Dataset", "SJR")

rf_results = pd.concat([assembly_rf, sjr_rf], ignore_index=True)
rf_results.insert(2, "Model", "Random Forest")

# Load XGBoost results
xgb_results = pd.read_csv(RESULTS_DIR / "xgboost_model_performance.csv")

# Combine both
all_results = pd.concat([rf_results, xgb_results], ignore_index=True)

# Select useful columns
all_results = all_results[
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

# Find best model for each target based on Test_R2
best_models = all_results.loc[
    all_results.groupby(["Dataset", "Target"])["Test_R2"].idxmax()
].reset_index(drop=True)

# Add quality label
def quality_label(r2):
    if r2 >= 0.90:
        return "Very good"
    elif r2 >= 0.75:
        return "Good"
    elif r2 >= 0.50:
        return "Medium"
    else:
        return "Weak"

best_models["Model_Quality"] = best_models["Test_R2"].apply(quality_label)

print("\nAll model comparison:")
print(all_results)

print("\nBest model for each target:")
print(best_models)

# Save files
all_results_file = RESULTS_DIR / "rf_vs_xgboost_all_results.csv"
best_models_file = RESULTS_DIR / "best_model_summary.csv"

all_results.to_csv(all_results_file, index=False)
best_models.to_csv(best_models_file, index=False)

print(f"\nAll comparison results saved to: {all_results_file}")
print(f"Best model summary saved to: {best_models_file}")

print("\nStep 12 completed successfully.")