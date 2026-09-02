# AECOPD-CV Risk Model

This repository contains the code, validation workflows, figures, and web calculator associated with the development and validation of the AECOPD-CV prediction model for cardiovascular complications among patients hospitalized with acute exacerbation of chronic obstructive pulmonary disease (AECOPD).

The repository includes the complete analytical framework used for model development, internal validation, external validation, pilot implementation evaluation, and sensitivity analyses.

## Repository Structure

### Internal Validation

Contains the scripts used for model development and internal validation within the derivation cohort, including model coefficient estimation, calibration assessment, discrimination analysis, decision curve analysis, and risk stratification.

### MIMIC-IV External Validation

Contains the scripts used for external validation of the frozen AECOPD-CV model within the MIMIC-IV database.

The following laboratory extraction strategies are included:

* 6-hour laboratory window (primary external validation analysis)
* 12-hour laboratory window
* 24-hour laboratory window
* Anytime laboratory extraction during hospitalization

Across all analyses, frozen model coefficients, frozen development-cohort median-imputation values, and frozen internally derived calibration parameters were applied without model refitting or external recalibration.

### eICU External Validation

Contains the scripts used for external validation of the frozen AECOPD-CV model within the eICU Collaborative Research Database.

The following laboratory extraction strategies are included:

* 6-hour laboratory window
* 12-hour laboratory window
* 24-hour laboratory window
* Anytime laboratory extraction during ICU admission

These analyses were performed to evaluate the robustness of model performance under varying conditions of laboratory availability within a heterogeneous multicentre ICU population.

### Independent Temporal Pilot Implementation

Contains the scripts used for independent temporal pilot implementation evaluation of the frozen AECOPD-CV model.

This evaluation was performed using a separate temporal cohort of hospitalized patients with AECOPD and was designed to assess model performance during pilot implementation under real-world clinical conditions. The implementation workflow includes patient-level prediction generation, discrimination analyses, risk-distribution visualizations, and publication-ready implementation figures.

No model refitting, recalibration, coefficient updating, or outcome-informed parameter estimation was performed.

### Imputation Sensitivity Analysis

Contains the scripts used to evaluate model robustness under alternative missing-data handling strategies, including:

* Median imputation
* Mean imputation
* K-nearest-neighbour imputation
* Iterative imputation
* Complete-case analysis

Performance was compared across discrimination, calibration, clinical utility, and risk-stratification metrics.

### Figures

Contains all figures generated during:

* Internal validation
* MIMIC-IV external validation
* eICU external validation
* Independent temporal pilot implementation evaluation
* Imputation sensitivity analyses

All figures were generated directly from reproducible scripted workflows and were not manually modified after generation.

### Web Calculator

Contains the standalone implementation of the AECOPD-CV risk calculator.

---

## Final Model Predictors

The final AECOPD-CV model includes six predictors:

* Age
* History of heart failure
* History of atrial fibrillation
* Blood gas pH
* Urea
* Lactate

---

## Outcome Definition

The primary outcome was a composite cardiovascular event occurring during hospitalization, defined as the occurrence of at least one of the following:

* Myocardial infarction
* Pulmonary embolism
* Pulmonary edema / acute heart failure decompensation phenotype
* Acute arrhythmia

---

## Data Sources

### Development and Internal Validation Cohort

Prospectively collected patients admitted with acute exacerbation of chronic obstructive pulmonary disease.

### Independent Temporal Pilot Implementation Cohort

A separate temporal cohort used for pilot implementation evaluation through structured medical-record review.

### External Validation Cohorts

* MIMIC-IV database
* eICU Collaborative Research Database

Access to both databases requires credentialed access through PhysioNet and completion of the required training and data-use agreements.

https://physionet.org/

---

## Reproducibility

The repository provides:

* Model-development scripts
* Internal-validation scripts
* MIMIC-IV external-validation scripts
* eICU external-validation scripts
* Independent temporal pilot implementation scripts
* Imputation sensitivity-analysis scripts
* Figure-generation workflows
* Patient-level prediction outputs
* Model coefficients
* Standalone web calculator

All analyses were implemented using fully reproducible scripted workflows.

---

# eICU Sensitivity Analysis: Alternative Median Imputation of Missing Laboratory Predictors

## Script Name

`eicu_imputation_sensitivity_6h_github_zenodo.py`

## Purpose

This script provides the reproducible implementation of an additional sensitivity analysis evaluating whether the performance of the selected eICU model was materially dependent on the specific values used to replace missing laboratory predictors.

The primary eICU analysis used predefined median-imputation values derived from the original model-development cohort. Because availability of pH, urea, and lactate was incomplete within the first 6 hours of ICU admission, and because the distributions of these laboratory variables differed between the development and eICU cohorts, an additional sensitivity analysis was performed in which missing laboratory measurements were instead replaced using medians calculated from the observed eICU measurements available within the same 6-hour window.

The purpose of this analysis was to evaluate robustness to the choice of laboratory-imputation values. It was not intended to define a new clinical prediction model or to replace the primary analysis.

## Selected eICU Model

The sensitivity analysis evaluates the selected eICU model using the following variables:

- Age
- History of heart failure
- History of atrial fibrillation
- pH
- Urea
- Lactate
- pH missingness indicator
- Urea missingness indicator
- Lactate missingness indicator
- Respiratory-support indicator

The selected model is implemented as an L2-penalized logistic regression model with standardized predictors and is evaluated using repeated stratified cross-validation.

The same modelling procedure is repeated under both imputation strategies so that the only intended analytical difference is the value assigned to an originally missing pH, urea, or lactate measurement.

## Imputation Strategies

Two strategies are compared.

### 1. Development-cohort median imputation

Missing laboratory measurements are replaced using the predefined development-cohort medians:

- pH: 7.39
- Urea: 34.05 mg/dL
- Lactate: 1.20 mmol/L

### 2. eICU 6-hour median imputation

Missing laboratory measurements are replaced using medians calculated from the observed eICU measurements obtained within the first 6 hours following ICU admission:

- pH: 7.33
- Urea: 47.08 mg/dL
- Lactate: 1.60 mmol/L

All patients are retained under both strategies. Observed laboratory measurements remain unchanged, and imputation is applied only when the corresponding laboratory predictor is missing.

## Analysis Framework

The following elements are kept identical between the two analyses:

- eICU cohort construction
- Composite cardiovascular outcome
- History of heart failure
- History of atrial fibrillation
- Observed laboratory measurements
- Respiratory-support definition
- Laboratory-missingness indicators
- 0–360 minute laboratory extraction window
- Earliest available laboratory-value selection
- Laboratory plausibility criteria
- Post-imputation clipping rules
- Predictor set
- Standardization procedure
- L2 logistic-regression specification
- Five-fold repeated stratified cross-validation
- Ten cross-validation repeats
- Random-state specification

The only intended difference between the two analyses is the set of values used to replace missing pH, urea, and lactate measurements.

## Laboratory Availability

Within the 4210-patient eICU cohort, before imputation:

- pH was available in 1655 patients (39.3%)
- Urea was available in 1444 patients (34.3%)
- Lactate was available in 765 patients (18.2%)
- All three laboratory predictors were available in 304 patients (7.2%)

All 4210 patients were retained for prediction after application of the corresponding imputation strategy.

## Performance Assessment

The two imputation strategies are compared using measures of discrimination, overall prediction error, calibration, clinical utility, and risk stratification.

The script calculates:

- Area under the receiver operating characteristic curve (AUC)
- Bootstrap-derived 95% confidence intervals for the AUC
- Average precision
- Bootstrap-derived confidence intervals for average precision
- Brier score
- Calibration intercept
- Calibration slope
- LOWESS-smoothed calibration curves
- Decision-curve analysis across threshold probabilities from 0.10 to 0.50
- Bootstrap confidence intervals for net benefit
- Observed cardiovascular-event rates across predicted-risk quartiles
- Wilson 95% confidence intervals for quartile-specific event rates

## Sensitivity Analysis Results

Performance was essentially unchanged when the development-cohort imputation values were replaced by the corresponding observed eICU 6-hour medians.

Using development-cohort median imputation:

- AUC: 0.700
- 95% CI: 0.684–0.716
- Brier score: 0.2085
- Calibration intercept: −0.011
- Calibration slope: 0.976
- Average precision: 0.577

Using eICU 6-hour median imputation:

- AUC: 0.698
- 95% CI: 0.682–0.714
- Brier score: 0.2087
- Calibration intercept: −0.011
- Calibration slope: 0.976
- Average precision: 0.576

The closely similar performance estimates indicate that the performance of the selected eICU model was not materially dependent on the particular fixed values used to replace missing pH, urea, and lactate measurements.

This analysis should be interpreted specifically as a sensitivity analysis of the laboratory-imputation strategy. It does not eliminate the underlying limitation associated with incomplete laboratory availability.

## Generated Figures

The script generates four individual comparison figures:

1. `figure_imputation_sensitivity_roc_6h.png`
2. `figure_imputation_sensitivity_dca_6h.png`
3. `figure_imputation_sensitivity_calibration_6h.png`
4. `figure_imputation_sensitivity_quartiles_6h.png`

These are additionally combined into:

5. `eICU_Imputation_Sensitivity_Composite_6h.png`

The composite figure contains:

- **Panel A:** receiver operating characteristic curves
- **Panel B:** decision-curve analysis
- **Panel C:** calibration curves
- **Panel D:** observed cardiovascular-event rates across predicted-risk quartiles

## Additional Output Files

The script generates:

1. `imputation_sensitivity_performance_6h.csv`
2. `imputation_sensitivity_values_6h.csv`
3. `imputation_sensitivity_lab_availability_6h.csv`
4. `imputation_sensitivity_calibration_6h.csv`
5. `imputation_sensitivity_decision_curve_6h.csv`
6. `imputation_sensitivity_quartiles_6h.csv`
7. `imputation_sensitivity_patient_level_6h.csv`
8. `eICU_imputation_sensitivity_6h.xlsx`
9. `imputation_sensitivity_summary_6h.json`

The Excel workbook contains separate worksheets for performance, imputation values, laboratory availability, calibration, risk quartiles, and decision-curve analysis.

The patient-level output contains predicted probabilities under both imputation strategies to support independent verification and reproducibility.

## How to Run

```bash
python eicu_imputation_sensitivity_6h_github_zenodo.py \
  --eicu-dir "path_to_eicu_folder" \
  --output "eicu_imputation_sensitivity_6h"
---

## Disclaimer

This repository and associated calculator are intended for research and educational purposes only. The AECOPD-CV model is designed to support clinical decision-making and should not replace clinical judgment, institutional protocols, or specialist evaluation.

---

## Citation

If you use this repository, please cite the associated manuscript and repository DOI.

---
