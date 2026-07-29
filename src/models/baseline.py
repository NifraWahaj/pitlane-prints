"""
baseline.py
-----------
Trains an XGBoost multiclass classifier on per-lap features.
Target: Driver (VER / HAM / ALO)

Key design decision: train/test split is stratified by RACE, not random.
Random split would leak laps from the same race into train and test,
making the model look better than it actually generalizes.
Since we only have one race right now, we do a stratified lap split
with shuffle=False to preserve lap ordering within each driver.

Usage:
    python src/models/baseline.py
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import xgboost as xgb


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_features(features_dir: str, race_tag: str) -> pd.DataFrame:
    path = os.path.join(features_dir, f"{race_tag}_features.csv")
    df = pd.read_csv(path)
    print(f"[load] {df.shape[0]} laps, {df['Driver'].nunique()} drivers")
    print(f"       Laps per driver:\n{df.groupby('Driver').size().to_string()}")
    return df


FEATURE_COLS = [
    "brake_duration_ratio",
    "throttle_smoothness",
    "full_throttle_ratio",
    "coasting_ratio",
    "gear_change_freq",
    "speed_at_throttle_lift",
    "mean_corner_speed",
    "speed_variance",
    "throttle_brake_overlap",
]


def prepare_xy(df: pd.DataFrame):
    """
    Returns X (feature matrix), y (integer labels), label_encoder, groups.
    NaNs are left in place — XGBoost handles them natively via
    its built-in missing value routing at each split.
    groups = one id per race (Season_Race), used for track-aware CV so
    we can measure whether the model generalizes across circuits rather
    than just memorizing this-track patterns.
    """
    le = LabelEncoder()
    X = df[FEATURE_COLS].values
    y = le.fit_transform(df["Driver"].values)
    print(f"\n[prep] Classes: {list(le.classes_)}  →  labels {list(range(len(le.classes_)))}")

    if "Race" in df.columns and "Season" in df.columns:
        groups = (df["Season"].astype(str) + "_" + df["Race"]).values
        print(f"[prep] {len(set(groups))} distinct races/groups for track-aware CV")
    else:
        groups = None

    return X, y, le, groups


def train_and_evaluate(X, y, le, groups=None, random_seed: int = 42):
    """
    Runs two flavours of cross-validation:

    1. Stratified K-fold (within-distribution) — the historical eval.
       Laps from the same race can appear in both train and val folds,
       so this measures "can the model spot a driver's style at all."

    2. Group K-fold by race (cross-circuit) — only used when we have
       multiple races. Each fold holds out entire races, so no lap
       from a held-out track leaks into training. This is the honest
       test of whether the fingerprint generalizes across circuits
       rather than memorizing track-specific patterns.
    """
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    print("\n[cv] Running 5-fold stratified cross-validation (within-distribution)...")
    y_pred_oof = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_prob_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

    acc = accuracy_score(y, y_pred_oof)
    print(f"\n[result] Stratified OOF accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print(f"[result] Random baseline (1/N): {1/len(le.classes_):.4f}  ({100/len(le.classes_):.1f}%)")
    print(f"\n[result] Classification report:")
    print(classification_report(y, y_pred_oof, target_names=le.classes_))

    cm = confusion_matrix(y, y_pred_oof)
    print(f"[result] Confusion matrix (rows=actual, cols=predicted):")
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)

    y_pred_group, y_prob_group = None, None
    if groups is not None and len(set(groups)) >= 5:
        n_groups = len(set(groups))
        n_splits = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_splits)

        print(f"\n[cv] Running {n_splits}-fold GROUP cross-validation (held-out circuits)...")
        y_pred_group = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict")
        y_prob_group = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict_proba")
        acc_group = accuracy_score(y, y_pred_group)
        print(f"\n[result] Cross-circuit (GroupKFold) accuracy: {acc_group:.4f}  ({acc_group*100:.1f}%)")
        print(f"[result] Classification report (cross-circuit):")
        print(classification_report(y, y_pred_group, target_names=le.classes_))
        cm_group = confusion_matrix(y, y_pred_group)
        print(f"[result] Confusion matrix — cross-circuit (rows=actual, cols=predicted):")
        print(pd.DataFrame(cm_group, index=le.classes_, columns=le.classes_))

        gap = acc - acc_group
        print(f"\n[result] Generalization gap (within-track − cross-track): {gap:.4f} "
              f"({gap*100:.1f} pts) — smaller is better; large gap means the model "
              f"leans on track-specific patterns rather than pure driving style.")
    else:
        print("\n[cv] Skipping cross-circuit GroupKFold — need data from 5+ races.")

    # Train final model on ALL data — this is what we save
    print("\n[train] Fitting final model on full dataset...")
    model.fit(X, y)

    return model, y_pred_oof, y_prob_oof, y_pred_group, y_prob_group, cm


def save_oof_predictions(df, le, y_pred_oof, y_prob_oof, y_pred_group, y_prob_group, features_dir):
    """
    Save every lap's genuinely out-of-fold prediction — the model that scored
    it never saw it during training. This is what the dashboard's "Blind
    Identification Challenge" should sample from, instead of asking the
    final model (fit on 100% of the data) to re-score laps it already
    memorized, which would make the demo unverifiable / trivially "rigged".
    """
    out = df[["Driver", "Race", "Season", "LapNumber", "LapTime_s"]].copy()

    out["oof_pred_intrack"] = le.inverse_transform(y_pred_oof)
    for i, cls in enumerate(le.classes_):
        out[f"oof_prob_intrack_{cls}"] = y_prob_oof[:, i]

    if y_pred_group is not None:
        out["oof_pred_crosscircuit"] = le.inverse_transform(y_pred_group)
        for i, cls in enumerate(le.classes_):
            out[f"oof_prob_crosscircuit_{cls}"] = y_prob_group[:, i]

    path = os.path.join(features_dir, "all_races_oof_predictions.csv")
    out.to_csv(path, index=False)
    print(f"\n[saved] Out-of-fold predictions → {path}")
    return path


def save_model(model, le, models_dir: str, race_tag: str):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{race_tag}_xgb_baseline.pkl")
    le_path    = os.path.join(models_dir, f"{race_tag}_label_encoder.pkl")
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    print(f"\n[saved] Model → {model_path}")
    print(f"[saved] LabelEncoder → {le_path}")
    return model_path, le_path


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi_df


def main():
    config     = load_config()
    race_tag   = "all_races"
    features_dir = config["data"]["features_dir"]
    models_dir   = os.path.join("outputs", "models")
    seed         = config["model"]["random_seed"]

    df = load_features(features_dir, race_tag)
    X, y, le, groups = prepare_xy(df)
    model, y_pred, y_prob, y_pred_group, y_prob_group, cm = train_and_evaluate(
        X, y, le, groups=groups, random_seed=seed
    )
    save_model(model, le, models_dir, race_tag)
    save_oof_predictions(df, le, y_pred, y_prob, y_pred_group, y_prob_group, features_dir)

    fi = get_feature_importance(model, FEATURE_COLS)
    print("\n── Feature importance (final model) ──")
    print(fi.to_string(index=False))

    # Save feature importance CSV for notebook
    fi_path = os.path.join(features_dir, f"{race_tag}_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"\n[saved] Feature importance → {fi_path}")


if __name__ == "__main__":
    main()