import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# FCBGA with Lid Project
# Step 13: Tune weak models using RandomizedSearchCV
#
# Targets:
#   Assembly:
#       - Warpage post lid attach
#   SJR:
#       - DeltaW_BGA
#       - DeltaW_bump
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def tune_model(df, target, dataset_name):
    print("\n" + "=" * 70)
    print(f"Tuning weak model: {dataset_name} - {target}")
    print("=" * 70)

    df.columns = df.columns.str.strip()

    target_columns = [target]
    input_columns = [col for col in df.columns if col not in target_columns]

    # Remove all other known target columns if they exist
    all_possible_targets = [
        "ELK stress",
        "Warpage Post UF cure",
        "Warpage post lid attach",
        "DeltaW_BGA",
        "DeltaW_bump"
    ]

    input_columns = [
        col for col in input_columns if col not in all_possible_targets
    ]

    # Remove constant columns
    constant_columns = []
    for col in input_columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)

    input_columns = [col for col in input_columns if col not in constant_columns]

    print("\nConstant columns removed:")
    for col in constant_columns:
        print("-", col)

    print("\nInput columns:")
    for col in input_columns:
        print("-", col)

    X = df[input_columns]
    y = df[target]

    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    print("\nCategorical columns:")
    for col in categorical_columns:
        print("-", col)

    print("\nNumerical columns:")
    for col in numerical_columns:
        print("-", col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------------
    # Model 1: Tuned Random Forest
    # --------------------------------------------------------
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42))
        ]
    )

    rf_param_grid = {
        "regressor__n_estimators": [200, 300, 500, 800],
        "regressor__max_depth": [None, 3, 5, 7, 10, 15],
        "regressor__min_samples_split": [2, 3, 5, 8, 10],
        "regressor__min_samples_leaf": [1, 2, 3, 4, 5],
        "regressor__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_grid,
        n_iter=40,
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=1
    )

    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    rf_train_pred = best_rf.predict(X_train)
    rf_test_pred = best_rf.predict(X_test)

    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_train_mse = mean_squared_error(y_train, rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, rf_test_pred)

    print("\nBest Random Forest parameters:")
    print(rf_search.best_params_)

    print("\nTuned Random Forest results:")
    print(f"Train R2: {rf_train_r2:.4f}")
    print(f"Test R2:  {rf_test_r2:.4f}")
    print(f"Train MSE: {rf_train_mse:.8f}")
    print(f"Test MSE:  {rf_test_mse:.8f}")

    # --------------------------------------------------------
    # Model 2: Tuned Extra Trees
    # --------------------------------------------------------
    et_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ExtraTreesRegressor(random_state=42))
        ]
    )

    et_param_grid = {
        "regressor__n_estimators": [200, 300, 500, 800],
        "regressor__max_depth": [None, 3, 5, 7, 10, 15],
        "regressor__min_samples_split": [2, 3, 5, 8, 10],
        "regressor__min_samples_leaf": [1, 2, 3, 4, 5],
        "regressor__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0]
    }

    et_search = RandomizedSearchCV(
        estimator=et_pipeline,
        param_distributions=et_param_grid,
        n_iter=40,
        scoring="r2",
        cv=5,
        random_state=42,
        n_jobs=1
    )

    et_search.fit(X_train, y_train)

    best_et = et_search.best_estimator_

    et_train_pred = best_et.predict(X_train)
    et_test_pred = best_et.predict(X_test)

    et_train_r2 = r2_score(y_train, et_train_pred)
    et_test_r2 = r2_score(y_test, et_test_pred)
    et_train_mse = mean_squared_error(y_train, et_train_pred)
    et_test_mse = mean_squared_error(y_test, et_test_pred)

    print("\nBest Extra Trees parameters:")
    print(et_search.best_params_)

    print("\nTuned Extra Trees results:")
    print(f"Train R2: {et_train_r2:.4f}")
    print(f"Test R2:  {et_test_r2:.4f}")
    print(f"Train MSE: {et_train_mse:.8f}")
    print(f"Test MSE:  {et_test_mse:.8f}")

    # --------------------------------------------------------
    # Select better tuned model
    # --------------------------------------------------------
    if rf_test_r2 >= et_test_r2:
        best_model_name = "Tuned Random Forest"
        best_model = best_rf
        best_test_pred = rf_test_pred
        best_train_r2 = rf_train_r2
        best_test_r2 = rf_test_r2
        best_train_mse = rf_train_mse
        best_test_mse = rf_test_mse
    else:
        best_model_name = "Tuned Extra Trees"
        best_model = best_et
        best_test_pred = et_test_pred
        best_train_r2 = et_train_r2
        best_test_r2 = et_test_r2
        best_train_mse = et_train_mse
        best_test_mse = et_test_mse

    print("\nBest tuned model:")
    print(best_model_name)
    print(f"Best Test R2: {best_test_r2:.4f}")

    # --------------------------------------------------------
    # Save actual vs predicted plot for best tuned model
    # --------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_test_pred, edgecolor="black", alpha=0.7)

    min_value = min(y_test.min(), best_test_pred.min())
    max_value = max(y_test.max(), best_test_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r--",
        label="Ideal prediction"
    )

    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"{best_model_name}: {dataset_name} - {target}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    safe_target = target.replace(" ", "_")
    figure_path = FIGURES_DIR / f"tuned_best_actual_vs_predicted_{dataset_name}_{safe_target}.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {figure_path}")

    return {
        "Dataset": dataset_name,
        "Target": target,
        "Best_Tuned_Model": best_model_name,
        "Train_R2": best_train_r2,
        "Test_R2": best_test_r2,
        "Train_MSE": best_train_mse,
        "Test_MSE": best_test_mse
    }


# ============================================================
# Assembly weak target
# ============================================================
assembly_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
assembly_df = pd.read_csv(assembly_file)

assembly_df = assembly_df.rename(columns={
    "Lid thicknes": "Lid thickness",
    "Warpage post lid atach": "Warpage post lid attach"
})

results = []

results.append(
    tune_model(
        df=assembly_df,
        target="Warpage post lid attach",
        dataset_name="Assembly"
    )
)


# ============================================================
# SJR weak/medium targets
# ============================================================
sjr_file = DATA_DIR / "cleaned_sjr_lid_data.csv"
sjr_df = pd.read_csv(sjr_file)

sjr_df = sjr_df.rename(columns={
    "lid thickness": "Lid thickness"
})

results.append(
    tune_model(
        df=sjr_df,
        target="DeltaW_BGA",
        dataset_name="SJR"
    )
)

results.append(
    tune_model(
        df=sjr_df,
        target="DeltaW_bump",
        dataset_name="SJR"
    )
)


# ============================================================
# Save tuned model results
# ============================================================
results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / "tuned_weak_model_results.csv"
results_df.to_csv(output_file, index=False)

print("\nFinal tuned weak-model results:")
print(results_df)

print(f"\nTuned model results saved to: {output_file}")
print("\nStep 13 completed successfully.")