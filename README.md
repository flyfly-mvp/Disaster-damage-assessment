# Disaster-damage-assessment
Disaster damage assessment method based on multiple machine learning methods
1. Overview
This workflow implements a machine learning–based framework for classifying rainstorm disaster economic loss levels in Zhejiang Province, China. It includes data preprocessing, multiple machine learning model training with Monte Carlo Cross-Validation, performance comparison, deviation analysis, feature importance analysis, SHAP-based feature contribution interpretation, and result visualization.

2. Data Preprocessing

Input Data
Source file: 训练4.0_filled.csv containing hazard, exposure, vulnerability, and environmental variables.
Target variable: CPI订正的直接经济损失（万元） – Direct economic loss adjusted by the Consumer Price Index (CPI).

Feature Mapping
Chinese feature names are mapped to descriptive English names for consistency. For example:
'累计降雨量' → 'Cumulative rainfall(mm)'
'人均GDP' → 'Per capita GDP(yuan)'
'Impervious_Surface_mean' → 'Impervious surface(%)'

Loss Level Classification
Economic loss values are discretized into four categories:

0–8000 (×10⁴ CNY): Minor Disaster

8000–15000 (×10⁴ CNY): Medium Disaster

15000–35000 (×10⁴ CNY): Major Disaster

≥35000 (×10⁴ CNY): Catastrophic Disaster
The categories are label-encoded for model input.

Feature Scaling
Min-Max normalization is applied to scale all features to the [0, 1] range, reducing the impact of differing magnitudes.

3. Machine Learning Models

The study uses five multi-class classification models with the following configurations:

Random Forest
n_estimators=200, max_depth=None, bootstrap=True, class_weight='balanced', random_state=42, n_jobs=-1.

LightGBM
objective='multiclass', num_class=4, learning_rate=0.1, n_estimators=300, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42.

XGBoost
objective='multi:softprob', num_class=4, learning_rate=0.1, n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric='mlogloss', tree_method='hist', random_state=42.

CatBoost
loss_function='MultiClass', iterations=500, depth=6, learning_rate=0.1, l2_leaf_reg=3.0, random_state=42, verbose=0.

MLP Neural Network
hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, random_state=42.

4. Validation Strategy

Monte Carlo Cross-Validation
Five independent 70% training / 30% testing splits are performed using random seeds [101, 201, 301, 401, 501].
Metrics are averaged over the five runs to obtain stable estimates.

Evaluation Metrics

Accuracy: Overall correct classification rate.

F1 Score: Harmonic mean of precision and recall (weighted for multi-class).

Cohen’s Kappa: Agreement beyond chance (0.41–0.60 interpreted as moderate).

ROC-AUC: Probability that a correct prediction is assigned higher confidence than an incorrect one.

5. Model Performance Visualization

Performance Curves
Line charts compare Accuracy, F1 Score, Kappa, and ROC-AUC for all models, sorted by Accuracy.

Deviation Statistics
Prediction biases are classified into “Bias Up” and “Bias Down” by 1–3 levels relative to the true class.
A stacked bar chart visualizes the proportion of each deviation type for every model.

6. Feature Importance Analysis

Method
XGBoost’s gain-based feature importance is calculated for each Monte Carlo split.
Importance scores are normalized and averaged across the five runs.

Visualization
Horizontal bar plots show the top 17 most important features ranked by relative importance.

7. SHAP Contribution Analysis

Method
SHAP (Shapley Additive Explanations) values are computed for XGBoost predictions on each split’s test set.
Absolute SHAP values are averaged over samples, classes, and runs.

Visualization
Stacked bar plots display the top 17 features, with colors representing contributions to each disaster level.

8. Correlation Heatmap

Pearson correlation coefficients are calculated between features and the target variable.

The upper triangle of the correlation matrix is masked for clarity.

Custom style settings include bold labels, Microsoft YaHei font, black grid lines, and a coolwarm colormap.

9. Output Files

The workflow produces the following files:

output_corr/pearson1_corr_heatmap.svg – Pearson correlation heatmap.

output_line/model_comparison_sorted_by_accuracy.svg – Model performance comparison.

output_line/eval_model_deviation_statistics_percent.svg – Deviation statistics.

feature_importance_barplots/XGBoostfeature_importance.svg – XGBoost feature importance.

xgboost_shap_top17_fixed_order.svg – SHAP contribution analysis.
