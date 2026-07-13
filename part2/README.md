PART 2 – SUPERVISED MACHINE LEARNING 

Objective:
Develop regression and classification models using cleaned_data.csv with proper preprocessing, leak-free scaling, model evaluation, regularization and statistical validation.

Dataset
- Features (X): All predictor columns.
- Regression Target: person_income
- Classification Target: loan_status

Preprocessing
- Label encoded: loan_grade (ordinal)
- One-hot encoded: person_home_ownership, loan_intent, cb_person_default_on_file
- 80/20 train-test split (random_state=42)
- StandardScaler fitted only on training data to prevent data leakage.

Linear Regression
- MSE: 16,830,453.39
- R²: 0.5860

Top coefficients:
1. loan_percent_income (+4310.95)
2. person_income (+2397.34)
3. person_home_ownership_RENT (-948.45)

Positive coefficients increase prediction, negative coefficients decrease prediction.

Ridge Regression
- MSE: 16,830,740.66
- R²: 0.5860

Ridge produced nearly identical performance, indicating little multicollinearity. Alpha controls regularization strength.

Class Imbalance
Original:
- Class 0: 78.1%
- Class 1: 21.9%

SMOTE balanced the training set before Logistic Regression.

Logistic Regression
Accuracy: 0.79
Precision: 0.512
Recall: 0.782
F1: 0.619
ROC-AUC: 0.8603

Precision = TP/(TP+FP)
Recall = TP/(TP+FN)

Recall is more important because missing a likely default is generally more costly.

Threshold Analysis
0.30 -> P=0.379 R=0.883 F1=0.531
0.40 -> P=0.442 R=0.841 F1=0.580
0.50 -> P=0.512 R=0.782 F1=0.619
0.60 -> P=0.588 R=0.707 F1=0.642 (Best)
0.70 -> P=0.666 R=0.598 F1=0.630

Regularization
C=1.0: Precision=0.5118 Recall=0.7824 F1=0.6188 AUC=0.8603
C=0.01: Precision=0.5126 Recall=0.7803 F1=0.6187 AUC=0.8600

Reducing C had negligible effect.

Bootstrap (500 samples)
Mean Difference = 0.000283
95% CI = [-0.000127, 0.000666]

The confidence interval includes zero, so the difference between C=1.0 and C=0.01 is not statistically significant.

Conclusion
Linear Regression explained about 58.6% of variance.
Logistic Regression achieved good discrimination (AUC≈0.86).
Threshold 0.60 gave the best F1.
Regularization had minimal impact.
Bootstrap confirmed no reliable difference between the two logistic models.

