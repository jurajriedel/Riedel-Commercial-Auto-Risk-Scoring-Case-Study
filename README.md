# Commercial Auto Risk Scoring Case Study

Case study submission for the ZNA Junior Data Scientist role.

## Approach

I started with an exploratory data analysis (`EDA.ipynb`) to understand the data and identify potential issues before modeling. The EDA showed that only around 12% of policies have a claim, that `total_loss_amount` is strongly right-skewed, and that some variables contain information that would not be available at prediction time. 
I also found inconsistent categorical values and multiple rows for the same insured, which I considered when preparing the modeling pipeline and validation strategy.

With only around 12% of policies having a claim, and a small number of large losses strongly affecting `total_loss_amount`, I decided to focus on predicting the probability of a claim rather than the loss amount itself.

I reviewed the original prototype notebook (`risk_scorer.ipynb`) and rewrote it as a more modular Python script (`Riedel_Juraj_case_study.py`). One of the main issues I found was that the prototype used post-outcome fields as model features — variables that are only available after a claim has been reported or resolved. This would cause target leakage, so these fields were removed from the final model.

## Key Assumptions

- I modeled risk as the probability of at least one claim, not claim frequency or severity.
- The data dictionary identifies four fields that are only available after a claim has been resolved, so I excluded them from the model. Some of these fields are populated in `score.csv`, but they would not normally be available when scoring a new policy, so using them would introduce data leakage.
- `has_safety_program` was also excluded because it is not present in `score.csv`, so the model would not have this information when making predictions.
-  `Rideshare`, which appears in score.csv but not in train.csv, is detected as an unseen category and explicitly routed to `Other`.
- `Agriculture` is not mentioned in the data dictionary and appears in only 8 rows of `train.csv`. I grouped it into `Other` rather than keeping it as its own category, since
  there isn't enough data for the model to learn anything reliable from it.
- For missing numeric values, I use the median plus a flag column marking that the value was originally missing.
## Target Definition

```
target = 1 if claim_count > 0 else 0   (binary, per policy row)
```

## Validation Approach

I used two things together here.

1. **Temporal holdout.** Policies are sorted by `snapshot_date`, and the most recent 20% are held out for testing.
2. **Grouped, stratified cross-validation.** Within the development set I used `StratifiedGroupKFold`, grouped by `insured_id`. The same company can show up more than once (different coverage lines, different years), so grouping keeps all of one company's rows on the same side of the split. 

About 96% of the holdout policies belong to a company already present in the development data. It's mostly a renewal book. So instead of one number, I broke holdout results
into three groups: overall, insureds already seen in development, and insureds new to the holdout period. 

For the model, I compared `LogisticRegression` (with balanced class weights) against `HistGradientBoostingClassifier`, using cross-validated ROC-AUC, PR-AUC, and Brier score.
Accuracy wasn't used because only around 12% of policies have a claim.

LogisticRegression won on both AUC and PR-AUC.
HistGradientBoosting had a better Brier score before calibration, but this metric also depends on how well the model's probabilities are calibrated. After calibration, the difference between the two models became much smaller, so it did not affect the final model choice.

I calibrated the best model with `CalibratedClassifierCV`, using the same grouped folds from cross-validation so a company can't leak between the calibration step's own train/validation split. Rather than picking a method upfront, the script fits both isotonic and Platt/sigmoid calibration on those folds and evaluates both on holdout, selecting whichever gives the lower Brier score. In this run sigmoid won, only by a small margin (Brier 0.099 vs. 0.099, with sigmoid slightly ahead on AUC and PR-AUC too).

## Results

```
2026-09-15 11:01:58,490|INFO|------------------------------------------------------------
2026-09-15 11:01:58,490|INFO|STEP 3 - Compare candidate models (grouped, stratified CV)
2026-09-15 11:01:58,490|INFO|------------------------------------------------------------
2026-09-15 11:02:04,235|INFO|  logreg_baseline  AUC=0.660 (+/-0.011)  PR-AUC=0.219  Brier=0.229
2026-09-15 11:02:04,236|INFO|  hist_gb          AUC=0.643 (+/-0.011)  PR-AUC=0.215  Brier=0.106
2026-09-15 11:02:04,236|INFO|-> Selected model: logreg_baseline
```

LogisticRegression was selected as the final model. Next, the script compares calibration methods on holdout:

```
2026-09-15 11:02:04,236|INFO|------------------------------------------------------------
2026-09-15 11:02:04,236|INFO|STEP 4 - Compare calibration methods & evaluate on holdout
2026-09-15 11:02:04,236|INFO|------------------------------------------------------------
2026-09-15 11:02:05,156|INFO|  isotonic  AUC=0.694  PR-AUC=0.221  Brier=0.099
2026-09-15 11:02:05,470|INFO|  sigmoid   AUC=0.696  PR-AUC=0.223  Brier=0.099
2026-09-15 11:02:05,470|INFO|-> Selected calibration method: sigmoid (lowest holdout Brier score)
2026-09-15 11:02:05,518|INFO|  overall              n= 3008 AUC=0.696 PR-AUC=0.223 Brier=0.099
2026-09-15 11:02:05,519|INFO|  insured seen in dev  n= 2889 AUC=0.695 PR-AUC=0.223 Brier=0.100
2026-09-15 11:02:05,519|INFO|  insured new          n=  119 AUC=0.722 PR-AUC=0.284 Brier=0.085
```

I also ran permutation importance on the calibrated model (`feature_importance.csv`).
`coverage_type` (APD vs. AL) came out as the strongest predictor.

## Known Limitations and Trade-offs

- APD and AL are combined into one model. EDA showed some differences between the two coverage types, and coverage_type was the most important feature. Separate models could be worth testing. I kept one model mainly to keep the solution simpler and have more data available for training.
- Only two model families got tested, `LogisticRegression` and `HistGradientBoosting`. 
- The new-insured holdout group is small, with only 119 policies. Because of this, the higher AUC should be interpreted with caution and does not necessarily mean that the model performs better on new customers.
- `Rideshare` policies are mapped to Other because this category was not present in the training data. Predictions for these policies should be treated with caution, as the model has no direct training examples for this business type.
- Feature importance is currently shown only at an overall level. `report_feature_importance()` ranks the features based on their overall importance and saves the results to `feature_importance.csv`. It does not explain individual predictions.

## How to Run

Needs Python 3.10+.

```bash
pip install pandas numpy scikit-learn matplotlib joblib
python Riedel_Juraj_case_study.py
```

It expects `train.csv` and `score.csv` in the working directory. The EDA notebook (`EDA.ipynb`) can be opened and run separately in Jupyter.

This was also run end-to-end on Databricks Free Edition — the script reads an optional `DATA_DIR` environment variable, so it can point at wherever the CSVs are uploaded there instead of the local working directory.

Running the script gives you:
- `predictions.csv` — one row per `policy_id` from `score.csv`, with a calibrated `risk_score`.
- `risk_model.joblib` — the fitted, calibrated model.
- `holdout_diagnostics.png` — ROC and calibration curves on holdout.
- `feature_importance.csv` — permutation importance per feature.

## Production Readiness Roadmap

These steps were not implemented, but would be good before moving the model into
production.

**Modeling & validation**
- Test additional models such as XGBoost, LightGBM, or CatBoost, and evaluate whether separate APD and AL models perform better than the current pooled approach.
- Re-validate the calibration approach (isotonic vs. sigmoid) on a larger and more recent dataset, rather than relying on a one-off comparison.
- Evaluate splitting APD/AL into separate models

**Explainability & fairness**
- Add per-prediction explanations (e.g. SHAP) so an individual policy's score can be justified, not just the aggregate feature ranking already produced.
- Check model performance across `state` and `business_type` for potential fairness issues.

**Monitoring & operations**
- Monitor feature and prediction drift and set up alerts for unexpected changes — this would spot things like the unseen `Rideshare` category automatically instead of manual EDA.
- Define a regular retraining process and compare the model against the existing `risk_score_external` benchmark. Store predictions and later outcomes to continuously evaluate model performance.

**Engineering & governance**
- Move the current script into a versioned production pipeline with automated data and schema checks.
- Add model documentation, access controls, audit logging, and the necessary actuarial and compliance review before deployment.

## AI Tool Disclosure

Claude (Anthropic) was used as an AI assistant throughout this project — to review the
original prototype and help identify the data leakage and validation issues discussed above,
and to support the development of the Python script and EDA notebook. It was also used to improve parts of the documentation and to 
discuss some of the technical decisions made during the project, though every suggestion was checked against the actual dataset and 
code before being incorporated. All final code, analysis, and conclusions in this submission were
reviewed and are understood by me.
