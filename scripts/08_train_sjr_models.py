import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# FCBGA with Lid Project
# Step 8: Train ML models for SJR dataset
# Targets:
#   1. DeltaW_BGA
#   2. DeltaW_bump
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Load SJR dataset
# ------------------------------------------------------------
data_file = DATA_DIR / "cleaned_sjr_lid_data.csv"
df = pd.read_csv(data_file)

df.columns = df.columns.str.strip()

# Rename column for consistency
df = df.rename(columns={
    "lid thickness": "Lid thickness"
})

# ------------------------------------------------------------
# Define targets
# ------------------------------------------------------------
target_columns = [
    "DeltaW_BGA",
    "DeltaW_bump"
]

input_columns = [col for col in df.columns if col not in target_columns]

# ------------------------------------------------------------
# Remove constant columns
# ------------------------------------------------------------
constant_columns = []

for col in input_columns:
    if df[col].nunique() == 1:
        constant_columns.append(col)

print("\nConstant input columns removed:")
for col in constant_columns:
    print("-", col)

input_columns = [col for col in input_columns if col not in constant_columns]

# ------------------------------------------------------------
# Identify numerical and categorical columns
# ------------------------------------------------------------
categorical_columns = df[input_columns].select_dtypes(include=["object", "string"]).columns.tolist()
numerical_columns = df[input_columns].select_dtypes(exclude=["object", "string"]).columns.tolist()

print("\nFinal input columns:")
for col in input_columns:
    print("-", col)

print("\nNumerical columns:")
for col in numerical_columns:
    print("-", col)

print("\nCategorical columns:")
for col in categorical_columns:
    print("-", col)

# ------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# ------------------------------------------------------------
# Train one model for each SJR target
# ------------------------------------------------------------
results = []

for target in target_columns:
    print("\n" + "=" * 60)
    print(f"Training model for target: {target}")
    print("=" * 60)

    X = df[input_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None
            ))
        ]
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2:  {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.8f}")
    print(f"Test MSE:  {test_mse:.8f}")

    results.append({
        "Target": target,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_MSE": train_mse,
        "Test_MSE": test_mse
    })

    # Actual vs predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, edgecolor="black", alpha=0.7)

    min_value = min(y_test.min(), y_test_pred.min())
    max_value = max(y_test.max(), y_test_pred.max())

    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Ideal prediction")

    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"Actual vs Predicted: {target}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    figure_name = f"sjr_actual_vs_predicted_{target}.png"
    figure_path = FIGURES_DIR / figure_name

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
results_df = pd.DataFrame(results)
results_file = RESULTS_DIR / "sjr_model_performance.csv"
results_df.to_csv(results_file, index=False)

print("\nSJR model performance summary:")
print(results_df)

print(f"\nPerformance results saved to: {results_file}")
print("\nStep 8 completed successfully.")