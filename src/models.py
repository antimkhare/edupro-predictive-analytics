"""
models.py
=========
Trains, evaluates, and exposes all five regression models for both
prediction targets (EnrollmentCount and CourseRevenue).

Usage
-----
    from src.models import ModelPipeline
    pipeline = ModelPipeline()
    pipeline.fit(merged)
    results = pipeline.evaluate()          # dict of DataFrames
    rev_pred = pipeline.predict_revenue(X_new)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.features import (
    FEATURE_COLS,
    FEATURE_LABELS,
    TARGET_ENROLLMENT,
    TARGET_REVENUE,
    get_X,
    get_y,
)


# ── Model registry ─────────────────────────────────────────────────────────────
def _build_models() -> dict:
    return {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Lasso Regression":   Lasso(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }


class ModelPipeline:
    """
    Trains all five regressors on both targets and stores evaluation artefacts.

    Attributes
    ----------
    results : dict
        Keys: 'EnrollmentCount', 'CourseRevenue'
        Values: dict with keys 'metrics', 'importances', 'best_model', 'best_name'
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, cv: int = 5):
        self.test_size    = test_size
        self.random_state = random_state
        self.cv           = cv
        self.results: dict = {}
        self._fitted      = False

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, merged: pd.DataFrame) -> "ModelPipeline":
        """Fit all models on both targets and compute evaluation metrics."""
        X = get_X(merged)

        for target in [TARGET_ENROLLMENT, TARGET_REVENUE]:
            y = get_y(merged, target)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            metrics_rows = []
            trained: dict = {}

            for name, model in _build_models().items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)
                cv_r2 = cross_val_score(
                    model, X, y, cv=self.cv, scoring="r2"
                ).mean()

                metrics_rows.append({
                    "Model":   name,
                    "MAE":     round(mae, 2),
                    "RMSE":    round(rmse, 2),
                    "R2":      round(r2, 4),
                    "CV_R2":   round(cv_r2, 4),
                })
                trained[name] = model

            metrics_df = (
                pd.DataFrame(metrics_rows)
                .sort_values("R2", ascending=False)
                .reset_index(drop=True)
            )

            best_name  = metrics_df.iloc[0]["Model"]
            best_model = trained[best_name]

            # Feature importance via Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            importances = (
                pd.DataFrame({"Feature": FEATURE_COLS, "Importance": rf.feature_importances_})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            importances["FeatureLabel"] = importances["Feature"].map(FEATURE_LABELS)

            self.results[target] = {
                "metrics":     metrics_df,
                "importances": importances,
                "best_model":  best_model,
                "best_name":   best_name,
                "X":           X,
                "y":           y,
                "trained":     trained,
            }

        self._fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit(merged) before predicting.")

    def predict_revenue(self, X_input: np.ndarray) -> float:
        """Predict course revenue using the best revenue model (Ridge)."""
        self._check_fitted()
        model = self.results[TARGET_REVENUE]["best_model"]
        return float(max(0, model.predict(X_input)[0]))

    def predict_enrollment(self, X_input: np.ndarray) -> int:
        """Predict enrollment count using the best enrollment model."""
        self._check_fitted()
        model = self.results[TARGET_ENROLLMENT]["best_model"]
        return int(max(0, round(model.predict(X_input)[0])))

    # ── Summary helpers ────────────────────────────────────────────────────────
    def best_summary(self) -> pd.DataFrame:
        """Return a one-row-per-target best-model summary."""
        self._check_fitted()
        rows = []
        for target, res in self.results.items():
            top = res["metrics"].iloc[0]
            rows.append({
                "Target":     target,
                "Best Model": res["best_name"],
                "MAE":        top["MAE"],
                "RMSE":       top["RMSE"],
                "R2":         top["R2"],
                "CV R2":      top["CV_R2"],
            })
        return pd.DataFrame(rows)

    def print_report(self):
        """Print a formatted evaluation report to stdout."""
        self._check_fitted()
        for target, res in self.results.items():
            print(f"\n{'='*60}")
            print(f"  Target: {target}")
            print(f"  Best model: {res['best_name']}")
            print(f"{'='*60}")
            print(res["metrics"].to_string(index=False))
            print(f"\n  Top-3 Feature Importances:")
            print(res["importances"].head(3)[["FeatureLabel", "Importance"]].to_string(index=False))


if __name__ == "__main__":
    from src.data_loader import build_merged
    merged = build_merged()
    pipeline = ModelPipeline()
    pipeline.fit(merged)
    pipeline.print_report()
    print("\n\nBest model summary:")
    print(pipeline.best_summary().to_string(index=False))
