import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# FCBGA with Lid Project
# Step 10: Feature importance for assembly and SJR models
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Function to train model and extract feature importance
# ------------------------------------------------------------
def train_and_get_importance(df, input_columns, target, dataset_name):
    X = df[input_columns]
    y = df[target]

    categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42
            ))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Get transformed feature names
    fitted_preprocessor = model.named_steps["preprocessor"]

    feature_names = []

    if categorical_columns:
        cat_encoder = fitted_preprocessor.named_transformers_["categorical"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_columns)
        feature_names.extend(cat_feature_names)

    feature_names.extend(numerical_columns)

    # Get feature importance values
    importances = model.named_steps["regressor"].feature_importances_

    importance_df = pd.DataFrame({
        "Dataset": dataset_name,
        "Target": target,
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Save CSV
    safe_target_name = target.replace(" ", "_").replace("/", "_")
    csv_path = RESULTS_DIR / f"feature_importance_{dataset_name}_{safe_target_name}.csv"
    importance_df.to_csv(csv_path, index=False)

    print(f"\nFeature importance for {dataset_name} - {target}:")
    print(importance_df)

    print(f"\nSaved CSV: {csv_path}")

    # Plot top features
    top_df = importance_df.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(top_df["Feature"], top_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance: {dataset_name} - {target}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"feature_importance_{dataset_name}_{safe_target_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {fig_path}")

    return importance_df


# ============================================================
# Assembly dataset
# ============================================================
assembly_file = DATA_DIR / "cleaned_fcbga_lid_data.csv"
assembly_df = pd.read_csv(assembly_file)

assembly_df.columns = assembly_df.columns.str.strip()

assembly_df = assembly_df.rename(columns={
    "Lid thicknes": "Lid thickness",
    "Warpage post lid atach": "Warpage post lid attach"
})

assembly_targets = [
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach"
]

assembly_input_columns = [
    col for col in assembly_df.columns if col not in assembly_targets
]

# Remove constant columns
assembly_input_columns = [
    col for col in assembly_input_columns if assembly_df[col].nunique() > 1
]

all_importance_results = []

for target in assembly_targets:
    importance_df = train_and_get_importance(
        df=assembly_df,
        input_columns=assembly_input_columns,
        target=target,
        dataset_name="Assembly"
    )
    all_importance_results.append(importance_df)


# ============================================================
# SJR dataset
# ============================================================
sjr_file = DATA_DIR / "cleaned_sjr_lid_data.csv"
sjr_df = pd.read_csv(sjr_file)

sjr_df.columns = sjr_df.columns.str.strip()

sjr_df = sjr_df.rename(columns={
    "lid thickness": "Lid thickness"
})

sjr_targets = [
    "DeltaW_BGA",
    "DeltaW_bump"
]

sjr_input_columns = [
    col for col in sjr_df.columns if col not in sjr_targets
]

# Remove constant columns
sjr_input_columns = [
    col for col in sjr_input_columns if sjr_df[col].nunique() > 1
]

for target in sjr_targets:
    importance_df = train_and_get_importance(
        df=sjr_df,
        input_columns=sjr_input_columns,
        target=target,
        dataset_name="SJR"
    )
    all_importance_results.append(importance_df)


# ============================================================
# Save all feature importance results together
# ============================================================
final_importance_df = pd.concat(all_importance_results, ignore_index=True)

final_importance_file = RESULTS_DIR / "all_feature_importance_results.csv"
final_importance_df.to_csv(final_importance_file, index=False)

print(f"\nAll feature importance results saved to: {final_importance_file}")
print("\nStep 10 completed successfully.")