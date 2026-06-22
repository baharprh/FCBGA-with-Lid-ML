import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# FCBGA with Lid Project
# Step 15: Create report-ready ML results table and plot
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load final best model summary
# ------------------------------------------------------------
summary_file = RESULTS_DIR / "final_best_model_summary_after_tuning.csv"
df = pd.read_csv(summary_file)

# ------------------------------------------------------------
# Round values for report readability
# ------------------------------------------------------------
report_df = df.copy()

report_df["Train_R2"] = report_df["Train_R2"].round(4)
report_df["Test_R2"] = report_df["Test_R2"].round(4)
report_df["Train_MSE"] = report_df["Train_MSE"].round(8)
report_df["Test_MSE"] = report_df["Test_MSE"].round(8)

# Reorder columns
report_df = report_df[
    [
        "Dataset",
        "Target",
        "Model",
        "Train_R2",
        "Test_R2",
        "Train_MSE",
        "Test_MSE",
        "Model_Quality"
    ]
]

print("\nReport-ready final model summary:")
print(report_df.to_string(index=False))

# Save clean report table
report_table_file = RESULTS_DIR / "report_ready_final_model_summary.csv"
report_df.to_csv(report_table_file, index=False)

print(f"\nReport-ready table saved to: {report_table_file}")

# ------------------------------------------------------------
# Create Test R2 bar plot
# ------------------------------------------------------------
plot_df = report_df.copy()
plot_df["Label"] = plot_df["Dataset"] + "\n" + plot_df["Target"]

plt.figure(figsize=(10, 6))
plt.bar(plot_df["Label"], plot_df["Test_R2"])

plt.axhline(y=0.90, linestyle="--", linewidth=1, label="Very good threshold")
plt.axhline(y=0.75, linestyle="--", linewidth=1, label="Good threshold")
plt.axhline(y=0.50, linestyle="--", linewidth=1, label="Medium threshold")

plt.ylabel("Test R²")
plt.xlabel("Prediction target")
plt.title("Final Best Model Performance for FCBGA with Lid Dataset")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

figure_file = FIGURES_DIR / "final_best_model_test_r2_summary.png"
plt.savefig(figure_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Final Test R2 plot saved to: {figure_file}")

# ------------------------------------------------------------
# Create a simple text summary file
# ------------------------------------------------------------
summary_text = f"""
Final ML Model Summary for FCBGA with Lid Project

The best model for each output was selected based on the highest test R2 value.

{report_df.to_string(index=False)}

Key observations:
1. ELK stress and warpage after UF cure were predicted with very good accuracy.
2. Post-lid-attach warpage remained weak even after tuning.
3. DeltaW_BGA improved after tuning and reached medium prediction quality.
4. DeltaW_bump remained weak, suggesting that the current input variables may not fully explain bump-level reliability behavior.
"""

text_file = RESULTS_DIR / "final_results_interpretation.txt"

with open(text_file, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"Text interpretation saved to: {text_file}")

print("\nStep 15 completed successfully.")