PART 2 – SUPERVISED MACHINE LEARNING 

Objective:
Develop regression and classification models using cleaned_data.csv with proper preprocessing, leak-free scaling, model evaluation, regularization and statistical validation.

Dataset
- Features (X): All predictor columns.
- Regression Target: person_income
- Classification Target: loan_status

Feature Matrix (X)

The feature matrix contains all predictor variables except the target variables used for regression and classification.

Excluded columns:

loan_amnt
loan_status
Regression Target (y_reg)

The regression target is loan_amnt, a continuous numeric variable representing the amount of money requested by an applicant. This target is appropriate for regression because it contains continuous values.

Classification Target (y_clf)

The classification target is loan_status, an existing binary variable in the dataset.

Typical values are:

0 → Loan Not Defaulted (or Approved)
1 → Loan Defaulted (or Rejected)

Since the dataset already provides a natural binary target, no additional binarization was required.



Preprocessing
- Label encoded: loan_grade (ordinal)
- One-hot encoded: person_home_ownership, loan_intent, cb_person_default_on_file
- 80/20 train-test split (random_state=42)
- StandardScaler fitted only on training data to prevent data leakage.

Categorical Feature Encoding

The feature matrix contained four categorical variables.

Ordinal Encoding

The loan_grade column was encoded using manual label encoding because its categories have a meaningful order representing increasing credit risk.

Grade	Encoded Value
A	1
B	2
C	3
D	4
E	5
F	6
G	7

This encoding preserves the natural ordering of loan grades while allowing machine learning algorithms to process the values numerically.

One-Hot Encoding

The remaining categorical variables:

person_home_ownership
loan_intent
cb_person_default_on_file

do not have a natural ordering. They were therefore encoded using One-Hot Encoding with drop_first=True.

Using one-hot encoding creates separate binary indicator columns for each category. Setting drop_first=True removes one dummy variable from each categorical feature, reducing multicollinearity (the dummy variable trap) while retaining all necessary information.

Unlike label encoding, one-hot encoding does not introduce an artificial ordinal relationship between categories. For example, assigning integers to home ownership categories (e.g., RENT = 1, OWN = 2, MORTGAGE = 3) would incorrectly imply that these categories have an inherent ranking, which is not true. One-hot encoding avoids this issue by representing each category independently.

Why Was the Scaler Fitted Only on the Training Data?

Fitting the scaler on the entire dataset before splitting would introduce data leakage because the calculated mean and standard deviation would include information from the test set. This allows information from the unseen test data to influence the preprocessing of the training data, leading to overly optimistic model performance estimates.

By fitting the scaler only on the training data, the test set remains completely unseen during preprocessing. This simulates how the model would perform on new, real-world data and provides a fair and unbiased evaluation of model performance.

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

Both Linear Regression and Ridge Regression achieved almost identical performance on the test dataset. This indicates that the dataset does not suffer from severe multicollinearity and that the ordinary least squares model already generalizes reasonably well.

Ridge Regression applies L2 regularization, which shrinks coefficient values to reduce overfitting. With alpha = 1.0, the regularization effect is relatively small, resulting in performance that is very similar to Linear Regression. Although Ridge did not improve the evaluation metrics in this case, it remains a valuable technique because it can produce more stable models when predictors are highly correlated.

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



---

# 1. Class Imbalance Analysis

## Output

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| 0     | 20,254 |     78.10% |
| 1     |  5,678 |     21.90% |

### Interpretation

The target variable `loan_status` is **imbalanced**, with the majority class (0) representing **78.10%** of the training data and the minority class (1) representing **21.90%**.

Since the minority class accounts for less than **35%** of the training samples, the dataset satisfies the assignment's condition for handling class imbalance.

To address this issue, **SMOTE (Synthetic Minority Oversampling Technique)** was applied **only to the training dataset**.

After SMOTE:

| Class |  Count |
| ----- | -----: |
| 0     | 20,254 |
| 1     | 20,254 |

This created a balanced training dataset without modifying the test set, ensuring an unbiased evaluation.

---

# 2. Confusion Matrix

```
                Predicted

               0        1

Actual 0     4020     1053

Actual 1      307     1104
```

### Interpretation

* **True Negatives (TN)** = **4020**
* **False Positives (FP)** = **1053**
* **False Negatives (FN)** = **307**
* **True Positives (TP)** = **1104**

The model correctly classified most loan applications.

More importantly, only **307** risky applicants were missed (false negatives), which is favorable because failing to identify risky loans is generally more costly than incorrectly flagging some safe applicants.

---

# 3. Classification Report

| Metric    | Class 0 | Class 1  |
| --------- | ------- | -------- |
| Precision | 0.93    | **0.51** |
| Recall    | 0.79    | **0.78** |
| F1-score  | 0.86    | **0.62** |

Overall Accuracy:

```
79%
```

---

## Accuracy

```
Accuracy = 79%
```

The model correctly predicts approximately **79%** of all loan applications.

---

## Precision

For the positive class:

```
Precision = 0.51
```

This means:

> Out of all applicants predicted as positive (class 1), **51%** were actually positive.

Because SMOTE encourages the model to detect more positives, it also increases the number of false positives, which lowers precision.

---

## Recall

```
Recall = 0.78
```

This is the strongest metric for your project.

It means:

> The model successfully identified **78%** of all actual positive cases.

For a loan approval or credit risk application, this is desirable because it reduces the number of risky applicants that go undetected.

---

## F1-score

```
F1 = 0.62
```

The F1-score balances Precision and Recall.

A value of **0.62** indicates a reasonable trade-off between correctly identifying risky applicants and limiting false alarms.

---

# 4. AUC

```
AUC = 0.8603
```

This is an excellent result.

### Interpretation

An **AUC of 0.8603** means that the model has an **86.03% probability of ranking a randomly selected positive instance higher than a randomly selected negative instance**.

AUC values can generally be interpreted as:

| AUC       | Interpretation    |
| --------- | ----------------- |
| 0.50      | No discrimination |
| 0.60–0.70 | Poor              |
| 0.70–0.80 | Fair              |
| 0.80–0.90 | **Good**          |
| >0.90     | Excellent         |

Model falls into the **Good** category, indicating strong class-separation ability.

---

# 5. Threshold Analysis

| Threshold | Precision |    Recall |        F1 |
| --------: | --------: | --------: | --------: |
|      0.30 |     0.379 | **0.883** |     0.531 |
|      0.40 |     0.442 |     0.841 |     0.580 |
|      0.50 |     0.512 |     0.782 |     0.619 |
|      0.60 | **0.588** |     0.707 | **0.642** |
|      0.70 | **0.666** |     0.598 |     0.630 |

---

## Best Threshold

The highest F1-score is:

```
Threshold = 0.60

F1 = 0.642
```

This threshold provides the best overall balance between Precision and Recall.

---

## What Happens as the Threshold Changes?

### Lower Threshold (0.30)

* Precision decreases
* Recall increases

The model predicts more applicants as positive, capturing more true positives but also increasing false positives.

---

### Higher Threshold (0.70)

* Precision increases
* Recall decreases

The model becomes more conservative, reducing false positives but missing more actual positive cases.

---

# Which Metric Is More Important?

For a **loan approval/default prediction** system:

**Recall** is generally the more important metric.

### Why?

A **false negative** means a risky applicant is incorrectly classified as low risk, potentially leading to financial losses for the lender.

Although higher recall may increase false positives, it reduces the likelihood of approving high-risk applicants, which is often the preferred trade-off in credit risk assessment.

---

# README Formulas

### Precision

[
\text{Precision} = \frac{TP}{TP + FP}
]

Precision measures the proportion of predicted positive cases that are actually positive.

---

### Recall

[
\text{Recall} = \frac{TP}{TP + FN}
]

Recall measures the proportion of actual positive cases that are correctly identified.




Conclusion
Linear Regression explained about 58.6% of variance.
Logistic Regression achieved good discrimination (AUC≈0.86).
Threshold 0.60 gave the best F1.
Regularization had minimal impact.
Bootstrap confirmed no reliable difference between the two logistic models.

