import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ============================================================
# FCBGA with Lid Project
# Step 16: Unified best-model training
#
# Combines the strongest practices from the original pipeline,
# UPDATE V1, and updatev2_depth:
#   - Cleaned CSV inputs with dataset-specific features
#   - Constant-column removal and column-name fixes
#   - No feature pooling across assembly/SJR files
#   - XGBoost for strong assembly targets
#   - Full RandomizedSearchCV for weak targets
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
ALL_TARGETS = {
    "ELK stress",
    "Warpage Post UF cure",
    "Warpage post lid attach",
    "DeltaW_BGA",
    "DeltaW_bump",
}


def quality_label(r2):
    if r2 >= 0.90:
        return "Very good"
    elif r2 >= 0.75:
        return "Good"
    elif r2 >= 0.50:
        return "Medium"
    else:
        return "Weak"


def load_assembly_data():
    df = pd.read_csv(DATA_DIR / "cleaned_fcbga_lid_data.csv")
    df.columns = df.columns.str.strip()
    return df.rename(
        columns={
            "Lid thicknes": "Lid thickness",
            "Warpage post lid atach": "Warpage post lid attach",
            "Bump solder material": "Cu-pillar bump solder material",
        }
    )


def load_sjr_data():
    df = pd.read_csv(DATA_DIR / "cleaned_sjr_lid_data.csv")
    df.columns = df.columns.str.strip()
    return df.rename(columns={"lid thickness": "Lid thickness"})


def prepare_features(df, target):
    input_columns = [col for col in df.columns if col not in ALL_TARGETS]
    constant_columns = [col for col in input_columns if df[col].nunique() == 1]
    input_columns = [col for col in input_columns if col not in constant_columns]

    categorical_columns = (
        df[input_columns].select_dtypes(include=["object", "string"]).columns.tolist()
    )
    numerical_columns = (
        df[input_columns].select_dtypes(exclude=["object", "string"]).columns.tolist()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", "passthrough", numerical_columns),
        ]
    )

    return input_columns, preprocessor


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    return {
        "Model": name,
        "Train_R2": r2_score(y_train, y_train_pred),
        "Test_R2": r2_score(y_test, y_test_pred),
        "Train_MSE": mean_squared_error(y_train, y_train_pred),
        "Test_MSE": mean_squared_error(y_test, y_test_pred),
        "estimator": pipeline,
        "y_test_pred": y_test_pred,
    }


def tune_candidates(dataset_name, target, df, candidates):
    print("\n" + "=" * 70)
    print(f"Unified training: {dataset_name} - {target}")
    print("=" * 70)

    input_columns, preprocessor = prepare_features(df, target)
    X = df[input_columns]
    y = df[target]

    print("\nInput columns:")
    for col in input_columns:
        print("-", col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidate_results = []

    if "xgb_assembly" in candidates:
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        xgb_search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions={
                "regressor__n_estimators": [200, 300, 400],
                "regressor__max_depth": [3, 4, 5, 6],
                "regressor__learning_rate": [0.03, 0.05, 0.08],
                "regressor__subsample": [0.8, 0.9, 1.0],
                "regressor__colsample_bytree": [0.8, 0.9, 1.0],
            },
            n_iter=24,
            scoring="r2",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        xgb_search.fit(X_train, y_train)
        print(f"\nBest XGBoost params: {xgb_search.best_params_}")
        candidate_results.append(
            evaluate_model(
                "XGBoost",
                xgb_search.best_estimator_,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    if "rf_tune" in candidates or "et_tune" in candidates:
        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
            ]
        )
        rf_param_grid = {
            "regressor__n_estimators": [200, 300, 500, 800],
            "regressor__max_depth": [None, 3, 5, 7, 10, 15],
            "regressor__min_samples_split": [2, 3, 5, 8, 10],
            "regressor__min_samples_leaf": [1, 2, 3, 4, 5],
            "regressor__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        }
        rf_search = RandomizedSearchCV(
            estimator=rf_pipeline,
            param_distributions=rf_param_grid,
            n_iter=40,
            scoring="r2",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        rf_search.fit(X_train, y_train)
        print(f"\nBest Random Forest params: {rf_search.best_params_}")
        candidate_results.append(
            evaluate_model(
                "Tuned Random Forest",
                rf_search.best_estimator_,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    if "et_tune" in candidates:
        et_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", ExtraTreesRegressor(random_state=RANDOM_STATE)),
            ]
        )
        et_param_grid = {
            "regressor__n_estimators": [200, 300, 500, 800],
            "regressor__max_depth": [None, 3, 5, 7, 10, 15],
            "regressor__min_samples_split": [2, 3, 5, 8, 10],
            "regressor__min_samples_leaf": [1, 2, 3, 4, 5],
            "regressor__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        }
        et_search = RandomizedSearchCV(
            estimator=et_pipeline,
            param_distributions=et_param_grid,
            n_iter=40,
            scoring="r2",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        et_search.fit(X_train, y_train)
        print(f"\nBest Extra Trees params: {et_search.best_params_}")
        candidate_results.append(
            evaluate_model(
                "Tuned Extra Trees",
                et_search.best_estimator_,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    if "xgb_sjr" in candidates:
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        xgb_search = RandomizedSearchCV(
            estimator=xgb_pipeline,
            param_distributions={
                "regressor__n_estimators": [100, 200, 300],
                "regressor__max_depth": [2, 3, 4, 5],
                "regressor__learning_rate": [0.03, 0.05, 0.08],
                "regressor__subsample": [0.7, 0.8, 0.9],
                "regressor__colsample_bytree": [0.7, 0.8, 0.9],
                "regressor__reg_lambda": [1.0, 2.0, 5.0],
            },
            n_iter=24,
            scoring="r2",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        xgb_search.fit(X_train, y_train)
        print(f"\nBest SJR XGBoost params: {xgb_search.best_params_}")
        candidate_results.append(
            evaluate_model(
                "XGBoost",
                xgb_search.best_estimator_,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )

    best = max(candidate_results, key=lambda row: row["Test_R2"])

    print("\nCandidate comparison:")
    for row in candidate_results:
        print(
            f"- {row['Model']}: Train R2={row['Train_R2']:.4f}, "
            f"Test R2={row['Test_R2']:.4f}"
        )
    print(f"\nSelected best model: {best['Model']}")

    safe_target = target.replace(" ", "_")
    figure_path = FIGURES_DIR / f"unified_best_actual_vs_predicted_{dataset_name}_{safe_target}.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best["y_test_pred"], edgecolor="black", alpha=0.7)
    min_value = min(y_test.min(), best["y_test_pred"].min())
    max_value = max(y_test.max(), best["y_test_pred"].max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Ideal prediction")
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"Unified Best Model: {dataset_name} - {target}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {figure_path}")

    return {
        "Dataset": dataset_name,
        "Target": target,
        "Model": best["Model"],
        "Train_R2": best["Train_R2"],
        "Test_R2": best["Test_R2"],
        "Train_MSE": best["Train_MSE"],
        "Test_MSE": best["Test_MSE"],
        "Model_Quality": quality_label(best["Test_R2"]),
    }


def main():
    assembly_df = load_assembly_data()
    sjr_df = load_sjr_data()

    target_plan = [
        ("Assembly", assembly_df, "ELK stress", ["xgb_assembly"]),
        ("Assembly", assembly_df, "Warpage Post UF cure", ["xgb_assembly"]),
        (
            "Assembly",
            assembly_df,
            "Warpage post lid attach",
            ["rf_tune", "et_tune"],
        ),
        ("SJR", sjr_df, "DeltaW_BGA", ["rf_tune"]),
        ("SJR", sjr_df, "DeltaW_bump", ["rf_tune", "xgb_sjr"]),
    ]

    best_rows = []

    for dataset_name, df, target, candidates in target_plan:
        result = tune_candidates(dataset_name, target, df, candidates)
        best_rows.append(result)

    best_df = pd.DataFrame(best_rows)
    best_output = RESULTS_DIR / "unified_best_model_summary.csv"
    best_df.to_csv(best_output, index=False)

    # Keep legacy filenames in sync for downstream scripts.
    legacy_output = RESULTS_DIR / "final_best_model_summary_after_tuning.csv"
    best_df.to_csv(legacy_output, index=False)

    print("\nUnified best model summary:")
    print(best_df.to_string(index=False))
    print(f"\nSaved: {best_output}")
    print(f"Saved: {legacy_output}")
    print("\nStep 16 completed successfully.")


if __name__ == "__main__":
    main()
