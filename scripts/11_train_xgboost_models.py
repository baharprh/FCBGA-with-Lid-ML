import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ============================================================
# FCBGA with Lid Project
# Step 11: Train XGBoost models for Assembly and SJR datasets
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def train_xgboost_models(df, target_columns, dataset_name):
    df.columns = df.columns.str.strip()

    input_columns = [col for col in df.columns if col not in target_columns]

    # Remove constant columns
    constant_columns = []
    for col in input_columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)

    print(f"\n{dataset_name}: Constant columns removed:")
    for col in constant_columns:
        print("-", col)

    input_columns = [col for col in input_columns if col not in constant_columns]

    X_all = df[input_columns]

    categorical_columns = X_all.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X_all.select_dtypes(exclude=["object", "string"]).columns.tolist()

    print(f"\n{dataset_name}: Final input columns:")
    for col in input_columns:
        print("-", col)

    print(f"\n{dataset_name}: Categorical columns:")
    for col in categorical_columns:
        print("-", col)

    print(f"\n{dataset_name}: Numerical columns:")
    for col in numerical_columns:
        print("-", col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns)
        ]
    )

    results = []

    for target in target_columns:
        print("\n" + "=" * 60)
        print(f"Training XGBoost model: {dataset_name} - {target}")
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
                ("regressor", XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42
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
            "Dataset": dataset_name,
            "Target": target,
            "Model": "XGBoost",
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

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            "r--",
            label="Ideal prediction"
        )

        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.title(f"XGBoost Actual vs Predicted: {dataset_name} - {target}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        safe_target = target.replace(" ", "_")
        figure_path = FIGURES_DIR / f"xgboost_actual_vs_predicted_{dataset_name}_{safe_target}.png"

        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved figure: {figure_path}")

    return pd.DataFrame(results)


# ============================================================
# Assembly dataset
# ============================================================
assembly_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
assembly_df = pd.read_csv(assembly_file)

assembly_df = assembly_df.rename(columns={
    "Lid thicknes": "Lid thickness",
    "Warpage post lid atach": "Warpage post lid attach"
})

assembly_targets = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach"
]

assembly_results = train_xgboost_models(
    df=assembly_df,
    target_columns=assembly_targets,
    dataset_name="Assembly"
)


# ============================================================
# SJR dataset
# ============================================================
sjr_file = DATA_DIR / "cleaned_sjr_lid_data.csv"
sjr_df = pd.read_csv(sjr_file)

sjr_df = sjr_df.rename(columns={
    "lid thickness": "Lid thickness"
})

sjr_targets = [
    "DeltaW_BGA",
    "DeltaW_bump"
]

sjr_results = train_xgboost_models(
    df=sjr_df,
    target_columns=sjr_targets,
    dataset_name="SJR"
)


# ============================================================
# Save XGBoost results
# ============================================================
all_results = pd.concat([assembly_results, sjr_results], ignore_index=True)

output_file = RESULTS_DIR / "xgboost_model_performance.csv"
all_results.to_csv(output_file, index=False)

print("\nXGBoost model performance summary:")
print(all_results)

print(f"\nXGBoost results saved to: {output_file}")
print("\nStep 11 completed successfully.")