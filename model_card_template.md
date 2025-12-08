# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Type:** Random Forest Classifier

**Version:** 0.0.1

**Date:** 12/08/2025

**Developer:** Machine Learning Team

**Model Architecture:** Ensemble learning method using multiple decision trees
- Algorithm: Random Forest (scikit-learn implementation)
- Number of estimators: 100 trees
- Max depth: 10 levels
- Random state: 42 (for reproducibility)
- Parallel processing: Enabled (n_jobs=-1)

**Input Features:** 
- Continuous: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- Categorical (one-hot encoded): workclass, education, marital-status, occupation, relationship, race, sex, native-country

**Output:** Binary classification predicting whether an individual's annual income exceeds $50K

## Intended Use

**Primary Use Cases:**
- Income prediction for demographic and economic research
- Educational tool for understanding machine learning pipelines in production
- Demonstration of bias detection and fairness evaluation in ML models

**Target Users:**
- Data scientists and ML practitioners learning production ML deployment
- Researchers analyzing income inequality patterns
- Students in machine learning courses

**Out-of-Scope Uses:**
- Making actual financial decisions about individuals
- Credit scoring or loan approval systems
- Any high-stakes decision-making without human oversight
- Real-time production systems without proper monitoring and validation

## Training Data

**Dataset:** UCI Census Income Dataset (1994 Census database)

**Source:** https://archive.ics.uci.edu/ml/datasets/census+income

**Size:** ~26,000 samples (80% of cleaned dataset after train-test split)

**Data Preprocessing:**
1. Removal of leading/trailing whitespace from all columns and values
2. Duplicate record removal
3. One-hot encoding of categorical features using scikit-learn's OneHotEncoder
4. Label binarization of target variable (<=50K vs >50K)

**Features:**
- **Demographics:** age, race, sex, native-country
- **Education:** education level, education-num (years of education)
- **Employment:** workclass, occupation, hours-per-week
- **Financial:** capital-gain, capital-loss, fnlgt (final weight)
- **Relationships:** marital-status, relationship

**Target Variable:** salary (<=50K or >50K annual income)

## Evaluation Data

**Dataset:** 20% hold-out test set from the UCI Census Income Dataset

**Size:** ~6,500 samples

**Preprocessing:** Same preprocessing pipeline as training data using fitted encoders

**Distribution:** Stratified sampling not explicitly applied; natural distribution maintained from original dataset split

## Metrics

**Primary Metrics:**

The model was evaluated using three standard classification metrics:

1. **Precision:** 0.7419
   - Interpretation: Of all individuals predicted to earn >50K, 74.19% actually earn >50K
   - Important for minimizing false positives

2. **Recall:** 0.6384
   - Interpretation: The model correctly identifies 63.84% of all individuals who actually earn >50K
   - Important for minimizing false negatives

3. **F-beta Score (F1):** 0.6861
   - Harmonic mean of precision and recall
   - Balanced measure of model performance

**Performance on Data Slices:**

Model performance varies across different demographic groups. Detailed slice analysis is available in `model/slice_output.txt`. Key observations:

- Performance varies by education level, with higher accuracy for individuals with advanced degrees
- Differences observed across occupation categories
- Variation in performance across demographic groups (race, sex) should be monitored for fairness

**Baseline Comparison:**
- Majority class baseline (always predict <=50K): ~75% accuracy but 0% recall for >50K class
- This model significantly outperforms naive baselines by actually learning patterns

## Ethical Considerations

**Bias and Fairness:**
- The model is trained on 1994 census data, which may not reflect current socioeconomic patterns
- Historical biases in the data (e.g., gender pay gaps, racial disparities) are likely encoded in the model
- Performance disparities across demographic groups have been documented in slice analysis
- The model should NOT be used for any decision-making that could reinforce existing inequalities

**Protected Attributes:**
- The model uses sensitive attributes (race, sex, native-country) as features
- This creates potential for discriminatory outcomes if used inappropriately
- Fair lending laws (ECOA, Fair Credit Reporting Act) prohibit use of such models in credit decisions

**Privacy:**
- Training data is publicly available census data (anonymized)
- No personally identifiable information (PII) is stored or processed
- The model should not be used to make predictions about specific, identifiable individuals

**Transparency:**
- Model architecture and hyperparameters are fully documented
- Training process and data preprocessing steps are transparent
- Slice performance analysis enables fairness auditing

## Caveats and Recommendations

**Limitations:**

1. **Temporal Validity:** Model is based on 1994 census data and may not generalize to current economic conditions
2. **Geographical Bias:** Primarily represents US population; not applicable to other countries
3. **Binary Classification:** Oversimplifies income distribution into two categories
4. **Class Imbalance:** Dataset is imbalanced toward lower-income class
5. **Feature Limitations:** Many relevant factors (debt, assets, family size) are not included

**Recommendations:**

1. **Regular Retraining:** Update model with recent census data when available
2. **Fairness Monitoring:** Continuously monitor performance across demographic groups
3. **Human Oversight:** Never use for automated decision-making without human review
4. **Bias Mitigation:** Consider fairness-aware training techniques for production deployment
5. **Threshold Tuning:** Adjust classification threshold based on specific use case requirements
6. **Ensemble Approach:** Consider combining with other models for more robust predictions
7. **Documentation:** Maintain detailed logs of model versions, data sources, and performance metrics

**Known Issues:**

- Model shows performance variation across demographic groups (see slice analysis)
- Precision-recall tradeoff favors precision over recall
- May underperform on edge cases and outliers
- Does not provide calibrated probability estimates

**Future Improvements:**

- Implement fairness constraints during training
- Explore feature engineering to reduce bias
- Test alternative algorithms (gradient boosting, neural networks)
- Develop separate models for different demographic segments
- Add model explainability tools (SHAP, LIME) for interpretability
