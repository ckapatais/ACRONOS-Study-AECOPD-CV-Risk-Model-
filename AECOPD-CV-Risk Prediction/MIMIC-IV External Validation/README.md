# MIMIC-IV External Validation

This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the MIMIC-IV database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the MIMIC-IV dataset without model refitting, recalibration, or coefficient updating. `predicted_probability_frozen` was generated using the original model coefficients, frozen development-cohort median imputation values, and prespecified internal calibration intercept/slope parameters derived before external validation. No external validation outcomes were used to estimate prediction parameters.

External validation analyses were performed using multiple laboratory extraction strategies:

- 6-hour laboratory window
- 12-hour laboratory window
- 24-hour laboratory window
- Anytime laboratory extraction window

For each extraction strategy, laboratory predictors were defined using the earliest available laboratory value within the corresponding time window following hospital admission.

## Validated Model Predictors

- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

## Composite Cardiovascular Outcome

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema / acute heart failure decompensation
- Acute arrhythmia

---

# Common Requirements

## Required Local Folder Structure

- Database.xlsx
- admissions.csv
- d_labitems.csv
- labevents.csv
- AECOPD_CV_FINAL_MODEL_Internal_Validation

The internal model directory must contain:

- model_coefficients_for_figures.csv
- recalibration_parameters.json
- analysis_dataset_internal_validation.csv

---

# MIMIC-IV External Validation: 6-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_6h.py`

## Output Files

1. lab_itemids_used_6h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_6h_labs.csv
3. Database_MIMIC-IV_external_validation_6h_labs_predictions.xlsx
4. lab_availability_6h.csv
5. outcome_definition_check_6h.csv
6. figure_roc_external_validation_6h.png
7. table_roc_external_validation_6h.csv
8. figure_decision_curve_external_validation_6h.png
9. table_decision_curve_external_validation_6h.csv
10. figure_calibration_external_validation_6h.png
11. table_calibration_external_validation_6h.csv
12. figure_quartiles_external_validation_6h.png
13. table_quartiles_external_validation_6h.csv
14. External_Validation_Composite_6h.png
15. summary_external_validation_6h.csv
16. summary_external_validation_6h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: 12-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_12h.py`

## Output Files

1. lab_itemids_used_12h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_12h_labs.csv
3. Database_MIMIC-IV_external_validation_12h_labs_predictions.xlsx
4. lab_availability_12h.csv
5. outcome_definition_check_12h.csv
6. figure_roc_external_validation_12h.png
7. table_roc_external_validation_12h.csv
8. figure_decision_curve_external_validation_12h.png
9. table_decision_curve_external_validation_12h.csv
10. figure_calibration_external_validation_12h.png
11. table_calibration_external_validation_12h.csv
12. figure_quartiles_external_validation_12h.png
13. table_quartiles_external_validation_12h.csv
14. External_Validation_Composite_12h.png
15. summary_external_validation_12h.csv
16. summary_external_validation_12h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: 24-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_24h.py`

## Output Files

1. lab_itemids_used_24h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_24h_labs.csv
3. Database_MIMIC-IV_external_validation_24h_labs_predictions.xlsx
4. lab_availability_24h.csv
5. outcome_definition_check_24h.csv
6. figure_roc_external_validation_24h.png
7. table_roc_external_validation_24h.csv
8. figure_decision_curve_external_validation_24h.png
9. table_decision_curve_external_validation_24h.csv
10. figure_calibration_external_validation_24h.png
11. table_calibration_external_validation_24h.csv
12. figure_quartiles_external_validation_24h.png
13. table_quartiles_external_validation_24h.csv
14. External_Validation_Composite_24h.png
15. summary_external_validation_24h.csv
16. summary_external_validation_24h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: Anytime Laboratory Extraction Window

## Script Name

`mimiciv_external_validation_anytime.py`

## Output Files

1. lab_itemids_used_anytime.csv
2. patient_level_external_validation_predictions_MIMIC-IV_anytime_labs.csv
3. Database_MIMIC-IV_external_validation_anytime_labs_predictions.xlsx
4. lab_availability_anytime.csv
5. outcome_definition_check_anytime.csv
6. figure_roc_external_validation_anytime.png
7. table_roc_external_validation_anytime.csv
8. figure_decision_curve_external_validation_anytime.png
9. table_decision_curve_external_validation_anytime.csv
10. figure_calibration_external_validation_anytime.png
11. table_calibration_external_validation_anytime.csv
12. figure_quartiles_external_validation_anytime.png
13. table_quartiles_external_validation_anytime.csv
14. External_Validation_Composite_anytime.png
15. summary_external_validation_anytime.csv
16. summary_external_validation_anytime.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV Sensitivity Analysis: Alternative Median Imputation of Missing Laboratory Predictors

## Script Name

`mimiciv_imputation_sensitivity_6h_github_zenodo.py`

## Purpose

This script was added to provide a fully reproducible sensitivity analysis evaluating whether the external-validation performance of the AECOPD-CV model in MIMIC-IV was materially dependent on the specific median values used to replace missing laboratory predictors.

The primary MIMIC-IV external-validation analysis used fixed median-imputation values derived from the original model-development cohort. Because availability of pH, urea, and lactate was incomplete within the early admission window, and because the distributions of these laboratory variables differed between the development and external-validation cohorts, an additional sensitivity analysis was performed using medians estimated directly from the observed MIMIC-IV laboratory measurements available within the first 6 hours of hospital admission.

The purpose of this analysis was not to develop or optimise a new prediction model. Instead, it was designed to determine whether the observed external-validation performance was sensitive to the particular fixed laboratory values used when pH, urea, or lactate measurements were missing.

## Analysis Framework

The analysis compares two laboratory-imputation strategies within the same MIMIC-IV validation cohort:

1. **Development-cohort median imputation**

   Missing pH, urea, and lactate measurements are replaced using the predefined median values derived from the original model-development cohort.

2. **MIMIC-IV 6-hour median imputation**

   Missing pH, urea, and lactate measurements are replaced using the corresponding medians calculated from the observed MIMIC-IV laboratory measurements available within the first 6 hours following hospital admission.

All patients are retained in both analyses. Patients with missing laboratory measurements are therefore not excluded from the sensitivity analysis. Observed laboratory values remain unchanged, and imputation is applied only when the corresponding predictor is missing.

The following components are kept identical between the two analyses:

- MIMIC-IV validation cohort
- Composite cardiovascular outcome
- Predictor definitions
- Age
- History of heart failure
- History of atrial fibrillation
- Observed pH measurements
- Observed urea measurements
- Observed lactate measurements
- Original regression coefficients
- Model intercept
- Prediction equation
- 6-hour laboratory extraction window

The only analytical difference between the two strategies is the set of values used to replace missing pH, urea, and lactate measurements.

No model refitting, coefficient updating, or dataset-specific optimisation is performed.

## Performance Assessment

Model performance under the two imputation strategies is compared using complementary measures of discrimination, overall prediction error, calibration, clinical utility, and risk stratification.

The script automatically calculates and exports:

- Area under the receiver operating characteristic curve (AUC)
- Bootstrap-derived 95% confidence intervals for the AUC
- Brier score
- Calibration intercept
- Calibration slope
- LOWESS-smoothed calibration curves
- Decision-curve analysis across threshold probabilities from 0.10 to 0.50
- Net-benefit estimates with bootstrap confidence intervals
- Observed cardiovascular-event rates across predicted-risk quartiles
- Wilson 95% confidence intervals for quartile-specific event rates

Both imputation strategies are displayed together in the graphical outputs to allow direct comparison of their predictive performance.

## Generated Figures

The script generates four individual comparison figures:

1. `figure_imputation_sensitivity_roc_6h.png`
2. `figure_imputation_sensitivity_dca_6h.png`
3. `figure_imputation_sensitivity_calibration_6h.png`
4. `figure_imputation_sensitivity_quartiles_6h.png`

These are additionally combined into a four-panel composite figure:

5. `MIMIC_IV_Imputation_Sensitivity_Composite_6h.png`

The composite figure contains:

- **Panel A:** receiver operating characteristic curves
- **Panel B:** decision-curve analysis
- **Panel C:** calibration curves
- **Panel D:** observed cardiovascular-event rates across predicted-risk quartiles

## Additional Output Files

The script also generates the following structured outputs:

1. `imputation_sensitivity_performance_6h.csv`
2. `imputation_sensitivity_values_6h.csv`
3. `imputation_sensitivity_lab_availability_6h.csv`
4. `imputation_sensitivity_calibration_6h.csv`
5. `imputation_sensitivity_decision_curve_6h.csv`
6. `imputation_sensitivity_quartiles_6h.csv`
7. `imputation_sensitivity_patient_level_6h.csv`
8. `MIMIC_IV_imputation_sensitivity_6h.xlsx`
9. `imputation_sensitivity_summary_6h.json`

The Excel workbook contains separate worksheets for overall performance, imputation values, laboratory availability, calibration, predicted-risk quartiles, and decision-curve analysis.

The patient-level output preserves the original analysis variables and includes the predicted probabilities obtained under both imputation strategies, allowing independent verification and reproducibility of the sensitivity analysis.

## Interpretation

This analysis specifically evaluates the robustness of the MIMIC-IV external-validation results to the choice of fixed laboratory-imputation values. Similar performance between the development-cohort median strategy and the MIMIC-IV-specific median strategy would indicate that the observed predictive performance is not materially dependent on the particular values used to replace missing pH, urea, and lactate measurements.

This sensitivity analysis does not remove the underlying limitation associated with incomplete laboratory availability in the external-validation cohort. It should therefore be interpreted as an assessment of robustness to the **choice of imputation values**, rather than as evidence that laboratory missingness itself is inconsequential.
---

# Notes

The primary MIMIC-IV external-validation scripts apply the predefined prediction framework using the original model parameters and development-cohort median-imputation values.

The additional 6-hour laboratory-imputation sensitivity analysis preserves the same cohort, outcome, predictor definitions, regression coefficients, and prediction equation while changing only the values used to replace missing pH, urea, and lactate measurements.

The sensitivity analysis is intended to assess robustness to the choice of laboratory-imputation values and should not be interpreted as development of an alternative prediction model.
