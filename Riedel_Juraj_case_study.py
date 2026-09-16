import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.inspection import permutation_importance
import os
DATA_DIR = os.environ.get("DATA_DIR", "")

logging.basicConfig(level=logging.INFO,format="%(asctime)s|%(levelname)s|%(message)s")
info=logging.getLogger("risk_scorer")

def log_section(title: str):
    info.info("")
    info.info("-" * 60)
    info.info(title)
    info.info("-" * 60)

def format_metrics(m: dict | None)->str:
    if m is None:
        return "n/a (too few rows or only one class present)"
    return f"n={m['n']:>5} AUC={m['auc']:.3f} PR-AUC={m['pr_auc']:.3f} Brier={m['brier']:.3f}"

num_features=[
    "vehicle_count","vehicle_avg_age","driver_count","driver_avg_age","years_in_business","prior_year_mileage_000","prior_apd_claim_count","prior_al_claim_count","prior_loss_amount",
    "deductible", "coverage_limit_000", "annual_premium","risk_score_external","num_heavy_vehicles", "late_payment_count","driver_per_vehicle", "heavy_vehicle_ratio", "loss_to_limit_ratio",]
cat_features=["coverage_type", "business_type_clean", "state"]
banned_columns=["claim_paid_amount_current_period","claim_status_current_period","days_to_first_claim_report","first_claim_reported_date","has_safety_program", "claim_count", "total_loss_amount",]

_KNOWN_BUSINESS_TYPES=None  

def load_and_prepare(path: str,is_train: bool)->pd.DataFrame:
    global _KNOWN_BUSINESS_TYPES
    df=pd.read_csv(path)
    info.info(f"Loaded file {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    df["business_type_clean"]= df["business_type"].astype(str).str.strip().str.lower().str.capitalize()
    df["business_type_clean"]= df["business_type_clean"].replace({"Agriculture": "Other"})
    if is_train:
        _KNOWN_BUSINESS_TYPES= set(df["business_type_clean"].unique())
    else:
        unseen= set(df["business_type_clean"].unique()) - _KNOWN_BUSINESS_TYPES
        if unseen:
            n_affected= df["business_type_clean"].isin(unseen).sum()
            info.warning(
                f"business_type categories not present in training data: {unseen} "
                f"({n_affected} rows in {path}) -> routing to 'Other' instead of "
                f"letting the encoder zero them out")
            df["business_type_clean"]= df["business_type_clean"].where(
                ~df["business_type_clean"].isin(unseen), "Other")
    df["driver_per_vehicle"]= df["driver_count"] / df["vehicle_count"].replace(0, np.nan)
    df["heavy_vehicle_ratio"]= df["num_heavy_vehicles"] / df["vehicle_count"].replace(0, np.nan)
    df["loss_to_limit_ratio"]= df["prior_loss_amount"] / (df["coverage_limit_000"].replace(0, np.nan) * 1000)

    for col in ["snapshot_date","policy_effective_date","policy_expiration_date"]:
        df[col]= pd.to_datetime(df[col], errors="coerce")
    if is_train:
        df["target"]= (df["claim_count"] > 0).astype(int)
        info.info(f"Positive rate (had a claim): {df['target'].mean():.3f}")
    return df


def validate_schema(train_df: pd.DataFrame, score_df: pd.DataFrame):
    all_features=num_features+cat_features
    offenders=set(banned_columns).intersection(all_features)
    if offenders:
        raise ValueError(f"Model is trying to use banned columns: {offenders}")
    missing_in_score= [c for c in all_features if c not in score_df.columns]
    if missing_in_score:
        raise ValueError(f"Features used in training are missing from score.csv: {missing_in_score}")
    missing_in_train = [c for c in all_features if c not in train_df.columns]
    if missing_in_train:
        raise ValueError(f"Features used are missing from train.csv: {missing_in_train}")

def get_preprocessor():
    num_transformer= Pipeline([
        ("imputer",SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler",StandardScaler()),])
    cat_transformer= Pipeline([
        ("imputer",SimpleImputer(strategy="constant", fill_value="missing")),("onehot",OneHotEncoder(handle_unknown="ignore")),])
    return ColumnTransformer([
        ("numeric_pipe",num_transformer,num_features),("categorical_pipe",cat_transformer,cat_features),])

def build_model():
    logreg= Pipeline([
        ("pre",get_preprocessor()),("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),])
    hgb= Pipeline([
        ("pre",get_preprocessor()),
        ("clf",HistGradientBoostingClassifier(
            random_state=42, max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,)),])
    return {"logreg_baseline": logreg,"hist_gb": hgb}


def time_holdout_split(df, holdout_frac=0.2):
    df_sorted= df.sort_values("snapshot_date")
    cutoff= df_sorted["snapshot_date"].quantile(1 - holdout_frac)
    dev= df_sorted[df_sorted["snapshot_date"]<=cutoff]
    holdout = df_sorted[df_sorted["snapshot_date"]>cutoff]
    info.info(f"Time split: dev={len(dev)} rows, holdout={len(holdout)} rows")
    return dev, holdout


def cross_validate(model,dev,n_splits=5):
    feature_cols= num_features + cat_features
    X, y, groups= dev[feature_cols], dev["target"], dev["insured_id"]
    sgkf= StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs,pr_aucs,briers = [], [], []
    for train_idx,val_idx in sgkf.split(X,y,groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        y_val = y.iloc[val_idx]
        aucs.append(roc_auc_score(y_val, proba))
        pr_aucs.append(average_precision_score(y_val, proba))
        briers.append(brier_score_loss(y_val, proba))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)), "brier_mean": float(np.mean(briers)),}

def evaluate_on_holdout(model, dev, holdout):
    feature_cols= num_features + cat_features
    X, y = holdout[feature_cols], holdout["target"]
    proba = model.predict_proba(X)[:, 1]
    seen_mask = holdout["insured_id"].isin(set(dev["insured_id"])).values

    def _metrics(mask):
        if mask.sum()<10 or y[mask].nunique()<2:
            return None
        return {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(y[mask], proba[mask])),
            "pr_auc": float(average_precision_score(y[mask], proba[mask])),
            "brier": float(brier_score_loss(y[mask], proba[mask])),
        }
    return {
        "overall": _metrics(np.ones(len(y), dtype=bool)),
        "insured_seen_in_dev": _metrics(seen_mask),
        "insured_new_in_holdout": _metrics(~seen_mask),
    }, proba, y

def save_diagnostic_plot(y_true, proba, path="holdout_diagnostics.png"):
    fig, axes= plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _= roc_curve(y_true, proba)
    axes[0].plot(fpr,tpr,label=f"AUC={roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("ROC - holdout")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()
    frac_pos, mean_pred= calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o", label="model")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes[1].set_title("Calibration - holdout")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed claim rate")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    info.info(f"Saved diagnostic plot to {path}")

def report_feature_importance(model, X, y, feature_cols, path="feature_importance.csv", top_n=10):
    result= permutation_importance(
        model, X, y, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    importance = (
        pd.Series(result.importances_mean, index=feature_cols)
        .sort_values(ascending=False)
        .rename("importance_mean")
    )
    importance.to_csv(path)
    info.info(f"Top {top_n} features by permutation importance (AUC drop):")
    for name, value in importance.head(top_n).items():
        info.info(f"  {name:<25s} {value:+.4f}")
    info.info(f"Full feature importance table saved to {path}")
    return importance

def select_calibration_method(model, dev, holdout, feature_cols, grouped_folds):
    results= {}
    fitted= {}
    for method in ["isotonic", "sigmoid"]:
        cal= CalibratedClassifierCV(model, method=method,cv=grouped_folds)
        cal.fit(dev[feature_cols], dev["target"])
        proba= cal.predict_proba(holdout[feature_cols])[:, 1]
        y= holdout["target"]
        results[method]= {
            "auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "brier": float(brier_score_loss(y, proba)),
        }
        fitted[method]= cal
        info.info(f"  {method:<9s} AUC={results[method]['auc']:.3f}  "
                   f"PR-AUC={results[method]['pr_auc']:.3f}  Brier={results[method]['brier']:.3f}")

    best_method = min(results, key=lambda m: results[m]["brier"])
    info.info(f"-> Selected calibration method: {best_method} (lowest holdout Brier score)")
    return best_method, fitted[best_method], results

def main():
    log_section("STEP 1 - Load & prepare data")
    train_df = load_and_prepare(f"{DATA_DIR}train.csv", is_train=True)
    score_df = load_and_prepare(f"{DATA_DIR}score.csv", is_train=False)
    validate_schema(train_df, score_df)
    info.info("Schema check passed: no banned columns, all features present in both files.")

    log_section("STEP 2 - Time-based train/holdout split")
    dev, holdout = time_holdout_split(train_df)

    log_section("STEP 3 - Compare candidate models (grouped, stratified CV)")
    candidates= build_model()
    cv_results= {name: cross_validate(model, dev) for name, model in candidates.items()}
    for name, res in cv_results.items():
        info.info(f"  {name:<16s} AUC={res['auc_mean']:.3f} (+/-{res['auc_std']:.3f})  "
                   f"PR-AUC={res['pr_auc_mean']:.3f}  Brier={res['brier_mean']:.3f}")
    best_name= max(cv_results, key=lambda n: cv_results[n]["auc_mean"])
    info.info(f"-> Selected model: {best_name}")

    log_section("STEP 4 - Compare calibration methods & evaluate on holdout")
    feature_cols= num_features + cat_features
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grouped_folds = list(sgkf.split(dev[feature_cols], dev["target"], dev["insured_id"]))

    cal_method, calibrated, cal_comparison = select_calibration_method(
        candidates[best_name], dev, holdout, feature_cols, grouped_folds
    )

    holdout_metrics, holdout_proba, holdout_y = evaluate_on_holdout(calibrated, dev, holdout)
    info.info(f"  overall              {format_metrics(holdout_metrics['overall'])}")
    info.info(f"  insured seen in dev  {format_metrics(holdout_metrics['insured_seen_in_dev'])}")
    info.info(f"  insured new          {format_metrics(holdout_metrics['insured_new_in_holdout'])}")
    save_diagnostic_plot(holdout_y, holdout_proba)
    report_feature_importance(calibrated, holdout[feature_cols], holdout["target"], feature_cols)

    log_section("STEP 5 - Refit on full training data & save model")
    sgkf_full = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    full_folds = list(sgkf_full.split(train_df[feature_cols], train_df["target"], train_df["insured_id"]))
    final_model = CalibratedClassifierCV(candidates[best_name], method=cal_method, cv=full_folds)
    final_model.fit(train_df[feature_cols], train_df["target"])

    joblib.dump(final_model, "risk_model.joblib")
    info.info("Saved fitted model to risk_model.joblib")

    log_section("STEP 6 - Score score.csv")
    score_risk = final_model.predict_proba(score_df[feature_cols])[:, 1]
    out = pd.DataFrame({"policy_id": score_df["policy_id"], "risk_score": score_risk})
    out.to_csv("predictions.csv", index=False)
    info.info(f"Wrote predictions for {len(out)} policies to predictions.csv")

    log_section("Summary")
    info.info(f"Selected model         : {best_name}")
    info.info(f"Calibration method      : {cal_method}")
    info.info(f"Holdout AUC / PR-AUC    : {holdout_metrics['overall']['auc']:.3f} / "
              f"{holdout_metrics['overall']['pr_auc']:.3f}")
    info.info(f"Holdout Brier           : {holdout_metrics['overall']['brier']:.3f}")
    info.info(f"Policies scored         : {len(out)}")
    info.info("Files created           : predictions.csv, risk_model.joblib, "
              "holdout_diagnostics.png, feature_importance.csv")

if __name__ == "__main__":
    main()
