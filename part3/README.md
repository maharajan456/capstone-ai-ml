
PART 3 – ADVANCED MODELING

Overview
This part extends the machine learning workflow by comparing multiple supervised learning algorithms, reducing overfitting, evaluating ensemble methods, tuning hyperparameters using GridSearchCV, building a reusable Scikit-learn Pipeline, analysing learning curves, and serializing the best-performing model for deployment.

------------------------------------------------------------
1. Decision Tree Baseline
------------------------------------------------------------
An unconstrained Decision Tree (max_depth=None) was trained.

Interpretation
- The model achieved 100% training accuracy while test accuracy was significantly lower.
- This indicates severe overfitting.
- Decision Trees are high-variance models because they greedily choose the locally best split without revisiting previous decisions.
- As the tree becomes deeper it memorizes the training data rather than learning general patterns.
Decision Tree Baseline

An unconstrained `DecisionTreeClassifier` was trained using the default `max_depth=None`. The model was evaluated on both the training and test datasets to assess its generalization performance and identify signs of overfitting.

Model Performance

| Metric                  |      Accuracy |
| ----------------------- | ------------: |
| Training Accuracy       | 1.0000 (100%) |
| Test Accuracy           |  0.8900 (89%) |
| Train-Test Accuracy Gap |  0.1100 (11%) |

Overfitting Analysis

The Decision Tree achieved a training accuracy of **100%**, indicating that it perfectly classified all observations in the training dataset. However, its test accuracy decreased to **89%**, producing an accuracy gap of **11 percentage points**.

This difference indicates that the unconstrained Decision Tree shows clear signs of **overfitting**. Since `max_depth=None`, the tree is allowed to grow deeply and create highly specific decision rules. As a result, the model may learn noise and patterns that are unique to the training dataset rather than only learning relationships that generalize to unseen data.

Although a test accuracy of 89% is relatively strong, the perfect training accuracy combined with the 11% train-test gap suggests that the model has high variance and does not generalize as consistently as its training performance suggests.

Why Decision Trees Are High-Variance Models

Decision Trees are considered high-variance models because they construct their structure greedily. At each node, the algorithm selects the split that provides the best immediate improvement according to the splitting criterion. Once a split is selected, the tree continues growing without revisiting and correcting earlier splitting decisions.

Small changes in the training dataset can therefore produce different early splits and lead to substantially different tree structures and predictions. An unconstrained tree can also continue creating branches until it captures highly specific training observations.

Conclusion

The baseline Decision Tree demonstrates overfitting because it achieved **100% training accuracy but only 89% test accuracy**, resulting in an **11% accuracy gap**. This confirms the high-variance nature of an unconstrained Decision Tree.

To improve robustness and reduce variance, ensemble techniques such as **Bagging** and **Random Forest** will be evaluated next. These methods combine multiple Decision Trees to produce more stable predictions and improve generalization to unseen data.


------------------------------------------------------------
2. Controlled Decision Tree
------------------------------------------------------------
A second tree was trained using:

max_depth = 5
min_samples_split = 20

Role of Hyperparameters

max_depth
- Limits tree depth.
- Reduces variance.
- Slightly increases bias.
- Improves generalization.

min_samples_split
- Prevents splitting nodes with very few observations.
- Avoids learning from noisy subsets.
- Produces smoother decision boundaries.

Comparison

The controlled tree produced a much smaller train-test accuracy gap than the unconstrained tree, demonstrating significantly better generalization.

Controlled Decision Tree

A second Decision Tree classifier was trained using `max_depth=5` and `min_samples_split=20`. These hyperparameters were introduced to control tree complexity and reduce the overfitting observed in the unconstrained Decision Tree.

Model Performance Comparison

| Model                       | Training Accuracy | Test Accuracy | Train-Test Gap |
| --------------------------- | ----------------: | ------------: | -------------: |
| Unconstrained Decision Tree |            1.0000 |        0.8900 |         0.1100 |
| Controlled Decision Tree    |            0.9117 |        0.9126 |        -0.0009 |

Interpretation of the Results

The unconstrained Decision Tree achieved **100% training accuracy** but only **89.00% test accuracy**, resulting in a train-test gap of approximately **11 percentage points**. This large performance gap indicates that the model overfit the training dataset. The tree was sufficiently complex to memorize training-specific patterns and noise, resulting in weaker generalization to unseen data.

In comparison, the controlled Decision Tree achieved **91.17% training accuracy** and **91.26% test accuracy**. The train-test gap was approximately **-0.09 percentage points**, which is substantially smaller than the 11% gap observed in the unconstrained model.

The slightly higher test accuracy than training accuracy is not a concern. The difference is extremely small and can occur because of normal sampling variation between the training and test datasets. It does not indicate overfitting.

Role of `max_depth`

The `max_depth` parameter limits how deep the Decision Tree can grow. Setting `max_depth=5` prevents the model from creating excessively deep branches and highly specific decision rules.

Restricting tree depth reduces model variance because the model is less sensitive to small variations and noise in the training dataset. However, this restriction can introduce additional bias because the model has less flexibility to learn very complex relationships.

Role of `min_samples_split`

The `min_samples_split` parameter specifies the minimum number of samples required for a node to be considered for splitting.

Setting `min_samples_split=20` prevents the model from creating additional splits based on small subsets of observations. Such splits may respond to random noise rather than meaningful patterns. Therefore, this parameter helps create a more stable and generalizable tree.

Comparison with the Unconstrained Tree

The train-test accuracy gap decreased from approximately **11% to almost 0%** after controlling the complexity of the Decision Tree. In addition, test accuracy improved from **89.00% to 91.26%**.

Although training accuracy decreased from **100% to 91.17%**, this reduction is beneficial because the model is no longer memorizing the training dataset. Instead, the controlled tree appears to capture more general patterns that transfer effectively to unseen data.

This demonstrates the **bias-variance trade-off**. The unconstrained Decision Tree had low training bias but high variance. Restricting the tree's complexity introduced some bias but substantially reduced variance and improved generalization.

Conclusion

The controlled Decision Tree performs better than the unconstrained Decision Tree in terms of generalization. Using `max_depth=5` and `min_samples_split=20` reduced the train-test accuracy gap from **11% to approximately 0%** while improving test accuracy from **89.00% to 91.26%**.

Therefore, the controlled Decision Tree is more robust and less prone to overfitting than the unconstrained baseline model. These results demonstrate that controlling model complexity can improve performance on unseen data even when training accuracy decreases.




------------------------------------------------------------
3. Gini vs Entropy
------------------------------------------------------------
Gini Impurity

Gini = 1 − Σ(pi²)

Entropy

Entropy = −Σ(pi log₂(pi))

Interpretation

A node with Gini = 0 means every sample belongs to exactly one class (perfect purity).

The entropy model achieved slightly higher accuracy than the Gini model, although the difference was very small. Both criteria are effective and typically produce similar trees.

Gini vs Entropy Comparison

Two Decision Tree classifiers were trained using max_depth=5 under the same training and testing conditions. The first model used criterion='gini', while the second model used criterion='entropy'.

Model Performance
Splitting Criterion	Test Accuracy
Gini	0.9126 (91.26%)
Entropy	0.9132 (91.32%)


Gini Impurity

Gini impurity measures the level of class mixing within a Decision Tree node.

The formula for Gini impurity is:

Gini = 1 − Σ pᵢ²

where pᵢ represents the proportion of samples belonging to class i within the node.

A node with Gini = 0 is a completely pure node. This means that all samples in the node belong to the same class. Therefore, there is no class mixture within the node.

Entropy

Entropy measures the uncertainty or disorder in the class distribution of a node.

The formula for Entropy is:

Entropy = −Σ pᵢ log₂(pᵢ)

where pᵢ represents the proportion of samples belonging to class i.

A lower entropy value represents a purer node, while a higher entropy value indicates greater class uncertainty. An entropy value of zero means that all samples in the node belong to one class.

Performance Interpretation

The Gini-based Decision Tree achieved a test accuracy of 91.26%, while the Entropy-based Decision Tree achieved a test accuracy of 91.32%.

Entropy produced the slightly higher test accuracy. However, the difference between the two models is only approximately 0.06 percentage points.

This difference is extremely small and does not provide strong evidence that Entropy is meaningfully better than Gini for this dataset. Both splitting criteria appear to identify very similar predictive patterns in the credit-risk data.

The result suggests that the choice between Gini impurity and Entropy has little practical impact on the predictive accuracy of the controlled Decision Tree for this problem.

Conclusion

The Entropy Decision Tree achieved the highest test accuracy of 91.32%, compared with 91.26% for Gini. Although Entropy technically produced the better result, the improvement is negligible.

Therefore, both criteria can be considered to have comparable performance on this dataset. The choice of splitting criterion is not a major factor affecting model performance in this experiment.

------------------------------------------------------------
4. Random Forest
------------------------------------------------------------
Random Forest combines many Decision Trees through bagging.

Top important features included:

• loan_percent_income
• loan_grade
• person_income
• person_home_ownership_RENT
• loan_int_rate

Feature Importance

Random Forest computes feature importance from the average reduction in Gini impurity contributed by each feature across every split of every tree.

Unlike Linear Regression coefficients, feature importance only measures predictive contribution and does not indicate whether the relationship is positive or negative.

------------------------------------------------------------
Bagging
------------------------------------------------------------
Each tree is trained on a bootstrap sample (sampling with replacement).

At every split only √(number of features) candidate variables are evaluated.

Averaging predictions from many independent trees reduces variance, improves stability and reduces overfitting compared with a single deep Decision Tree.

------------------------------------------------------------
Gradient Boosting
------------------------------------------------------------
Gradient Boosting builds trees sequentially.

Each new tree attempts to correct errors made by previous trees.

It generally achieved the strongest predictive performance among all evaluated models.

------------------------------------------------------------
Feature Ablation Study
------------------------------------------------------------
The five least important features were removed and a second Random Forest was trained.

Interpretation

The reduced model produced a slightly lower ROC-AUC than the full model.

This indicates that although these features had low individual importance, they still contributed useful predictive information.

Removing them simplifies deployment and reduces computational cost, but the small performance loss suggests retaining all features for production.
Random Forest Classification

A Random Forest classifier was trained using `n_estimators=100`, `max_depth=10`, and `random_state=42`.

Random Forest Performance

| Metric            |  Score |
| ----------------- | -----: |
| Training Accuracy | 0.9381 |
| Test Accuracy     | 0.9308 |
| ROC-AUC           | 0.9306 |

The Random Forest achieved a training accuracy of **93.81%** and a test accuracy of **93.08%**. The difference between training and test accuracy is approximately **0.73 percentage points**, indicating a small generalization gap.

Compared with the unconstrained Decision Tree, which achieved 100% training accuracy and 89.00% test accuracy, the Random Forest demonstrates substantially better generalization. The small train-test gap suggests that averaging predictions across multiple trees successfully reduces the variance associated with a single Decision Tree.

The ROC-AUC score of **0.9306** indicates that the Random Forest has strong ability to distinguish between the two `loan_status` classes.

Top Five Important Features

| Feature                    | Importance |
| -------------------------- | ---------: |
| loan_percent_income        |   0.295855 |
| loan_grade                 |   0.176031 |
| person_income              |   0.126358 |
| person_home_ownership_RENT |   0.114634 |
| loan_int_rate              |   0.084157 |

The most important feature was `loan_percent_income`, with an importance score of approximately **0.296**. This suggests that the proportion of a person's income represented by the loan amount plays an important role in the Random Forest's predictions'.

`loan_grade`, `person_income`, rental home ownership status, and `loan_int_rate` were also among the most influential features.

Random Forest feature importance is based on the average reduction in node impurity produced by splits involving each feature across the trees in the forest. With the default Gini criterion, features that repeatedly produce large reductions in Gini impurity receive higher importance scores.

A Random Forest feature importance score differs from a Linear Regression coefficient. A regression coefficient describes the direction and magnitude of the association between a feature and the predicted numeric value. For example, a positive coefficient indicates an increase in the predicted value as the feature increases, while holding other variables constant.

Random Forest feature importance does not directly describe a positive or negative relationship. Instead, it represents how useful a feature was for creating predictive splits. Therefore, an importance score of 0.295855 does not mean that `loan_percent_income` increases the prediction by 0.295855 units.

Bagging Concept in Random Forest

Random Forest is an ensemble learning method based on bagging. During training, each Decision Tree receives a bootstrap sample of the training dataset. Bootstrap sampling selects observations randomly with replacement, meaning that some observations may appear multiple times while others may not appear in a particular tree's training sample.

In addition, at each split, the Random Forest considers a random subset of features. For classification, the default number of features considered is approximately the square root of the total number of available features.

These sources of randomness encourage individual trees to learn different patterns. The final prediction is produced by combining the predictions of multiple trees. Ensemble averaging reduces variance and makes the Random Forest less sensitive to noise or individual training observations compared with a single deep Decision Tree.

Gradient Boosting Classification

A Gradient Boosting classifier was trained using `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, and `random_state=42`.

Gradient Boosting Performance

| Metric            |  Score |
| ----------------- | -----: |
| Training Accuracy | 0.9260 |
| Test Accuracy     | 0.9269 |
| ROC-AUC           | 0.9336 |

The Gradient Boosting model achieved a training accuracy of **92.60%** and a test accuracy of **92.69%**. The extremely small difference between training and test accuracy suggests good generalization and no clear evidence of overfitting.

The model achieved a ROC-AUC score of **0.9336**, which is slightly higher than the Random Forest ROC-AUC of **0.9306**.

Random Forest achieved slightly higher test accuracy at **93.08%**, compared with **92.69% for Gradient Boosting**. However, Gradient Boosting achieved better ROC-AUC.

For this credit-risk classification problem, ROC-AUC is particularly useful because the `loan_status` classes are imbalanced. The higher ROC-AUC suggests that Gradient Boosting provides slightly better overall class-separation ability across different classification thresholds.

Gradient Boosting trains trees sequentially. Each new tree attempts to correct errors made by the existing ensemble. The `learning_rate=0.1` parameter controls the contribution of each tree, while `n_estimators=100` specifies the number of boosting stages.

Feature Ablation Study

The five features with the lowest Random Forest importance scores were removed from the training and test feature matrices. A second Random Forest classifier was then trained using identical hyperparameters and `random_state=42`.

Ablation Results

| Model                 | Number of Features |  ROC-AUC |
| --------------------- | -----------------: | -------: |
| Full Random Forest    |                 17 | 0.930578 |
| Reduced Random Forest |                 12 | 0.926395 |

The full Random Forest achieved a ROC-AUC of **0.930578**, while the reduced Random Forest achieved a ROC-AUC of **0.926395**.

The ROC-AUC decreased by approximately **0.00418**, or about **0.42 percentage points**, after removing the five lowest-importance features.

This result suggests that the removed features were not completely uninformative. Although their individual feature importance scores were low, they collectively contributed a small amount of predictive information to the Random Forest.

Therefore, the ablation experiment does not indicate that these features were simply adding noise. Removing them caused a measurable, although relatively small, reduction in class-separation performance.

Production Implications

The reduced model uses **12 features instead of 17**, representing a reduction of approximately **29% in feature dimensionality**. A lower-dimensional model may provide production advantages, including lower feature-processing requirements, reduced inference cost, and a smaller feature maintenance burden.

However, model simplification resulted in an ROC-AUC decrease of approximately **0.00418**. Whether this trade-off is acceptable depends on the production system's tolerable performance threshold.

If minimizing infrastructure and feature-maintenance complexity is a priority and an AUC degradation of approximately 0.004 is acceptable, the reduced model may be considered. If maximum predictive performance is the primary objective, the full 17-feature Random Forest should be retained.

Overall Conclusion

The Random Forest achieved strong generalization with a test accuracy of **93.08%** and ROC-AUC of **0.9306**. The small train-test accuracy gap indicates that ensemble averaging successfully reduced the high variance observed in the unconstrained Decision Tree.

Gradient Boosting achieved the highest ROC-AUC of **0.9336**, suggesting slightly stronger class-separation ability than the Random Forest.

The feature ablation study demonstrated that the five lowest-importance Random Forest features still provided a small amount of collective predictive information. Removing them reduced ROC-AUC from **0.930578 to 0.926395**.

Based on the current test-set results, Gradient Boosting is the strongest model in terms of ROC-AUC, while Random Forest provides slightly higher classification accuracy. Cross-validation should be used next to determine whether these performance differences remain consistent across multiple data splits before selecting the final production model.


------------------------------------------------------------
5. Cross Validation
------------------------------------------------------------
Models evaluated

• Logistic Regression
• Controlled Decision Tree
• Random Forest
• Gradient Boosting

Cross-validation is more reliable than a single train-test split because:

• every observation is used for both training and validation
• performance is averaged across five independent folds
• estimates are less sensitive to random sampling
• variance of evaluation decreases

Cross-Validated Model Comparison

A five-fold stratified cross-validation procedure was performed to compare four classification models: Logistic Regression, Controlled Decision Tree, Random Forest, and Gradient Boosting.

`StratifiedKFold` was configured with `n_splits=5`, `shuffle=True`, and `random_state=42`. Stratification was used to preserve approximately the same `loan_status` class distribution within each fold.

ROC-AUC was selected as the evaluation metric because the target variable is imbalanced. ROC-AUC measures a model's ability to distinguish between the two classes across different classification thresholds.

#Cross-Validation Results

| Model                    | Mean ROC-AUC | Standard Deviation |
| ------------------------ | -----------: | -----------------: |
| Logistic Regression      |       0.8638 |             0.0029 |
| Controlled Decision Tree |       0.8913 |             0.0038 |
| Random Forest            |       0.9260 |             0.0030 |
| Gradient Boosting        |       0.9280 |             0.0022 |

### Logistic Regression

Logistic Regression achieved a mean ROC-AUC of **0.8638** with a standard deviation of **0.0029**.

The relatively low standard deviation indicates that the model performs consistently across the five validation folds. However, its mean ROC-AUC is lower than the tree-based ensemble models.

This suggests that Logistic Regression may be limited by its primarily linear decision boundary and may not fully capture the complex and non-linear relationships present in the credit-risk dataset.

### Controlled Decision Tree

The Controlled Decision Tree achieved a mean ROC-AUC of **0.8913** with a standard deviation of **0.0038**.

Its performance is higher than Logistic Regression, suggesting that the Decision Tree captures non-linear relationships and feature interactions more effectively.

However, the Decision Tree has the highest standard deviation among the four models. Although the variation is still small, this indicates slightly greater sensitivity to changes in the training and validation data.

### Random Forest

Random Forest achieved a mean ROC-AUC of **0.9260** with a standard deviation of **0.0030**.

The substantial improvement over the single Decision Tree demonstrates the advantage of ensemble learning. By combining predictions from multiple Decision Trees trained on bootstrap samples and random feature subsets, Random Forest reduces the variance associated with an individual tree.

The high mean ROC-AUC indicates strong class-separation ability, while the low standard deviation demonstrates consistent performance across the five validation folds.

### Gradient Boosting

Gradient Boosting achieved the highest mean ROC-AUC of **0.9280** and the lowest standard deviation of **0.0022**.

The high mean ROC-AUC indicates that Gradient Boosting provides the strongest overall ability to distinguish between the two `loan_status` classes among the evaluated models.

In addition, the lowest standard deviation indicates that its performance is highly consistent across different validation subsets.

Gradient Boosting builds trees sequentially, with each new tree attempting to correct errors made by the existing ensemble. This sequential error-correction process allows the model to capture complex patterns and feature interactions within the credit-risk dataset.

### Why Cross-Validation Is More Reliable

A single train-test split evaluates a model using only one specific division of the dataset. The resulting performance score may be influenced by the particular observations assigned to the training and test sets.

Five-fold cross-validation evaluates each model across five different validation subsets. During each iteration, four folds are used for model training and the remaining fold is used for validation. The process is repeated until every fold has served as the validation set.

The mean ROC-AUC represents the model's average generalization performance across multiple data splits. The standard deviation measures the consistency of the model's performance between folds.

Therefore, cross-validation provides a more reliable estimate of generalization performance than a single train-test split because the model is evaluated on multiple subsets of unseen data rather than one test split alone.

### Overall Model Comparison

The cross-validation results show a clear improvement as model complexity and ensemble learning techniques are introduced.

Logistic Regression produced a mean ROC-AUC of **0.8638**, while the Controlled Decision Tree improved the score to **0.8913**. Random Forest further increased the mean ROC-AUC to **0.9260**.

Gradient Boosting achieved the highest mean ROC-AUC of **0.9280** and the lowest standard deviation of **0.0022**.

The difference between Gradient Boosting and Random Forest is relatively small at approximately **0.002 ROC-AUC points**. Therefore, the results do not indicate an extremely large performance advantage for Gradient Boosting. However, Gradient Boosting consistently achieved the strongest average performance and the lowest variation across the five folds.

### Conclusion

Based on the five-fold cross-validation results, **Gradient Boosting is the strongest model among the four evaluated classifiers**.

It achieved the highest mean ROC-AUC of **0.9280** and the lowest standard deviation of **0.0022**, indicating both strong predictive performance and consistent generalization across different validation subsets.

Random Forest remains a strong alternative with a mean ROC-AUC of **0.9260**. However, Gradient Boosting is selected as the current best-performing model based on its slightly higher average ROC-AUC and lower cross-validation variability.

The cross-validation results support the earlier test-set evaluation, where Gradient Boosting also achieved the highest ROC-AUC. Therefore, Gradient Boosting is the leading candidate for the final production pipeline, subject to the remaining model tuning and production evaluation steps.



------------------------------------------------------------
6. Hyperparameter Tuning
------------------------------------------------------------
Parameter Grid

n_estimators:
50,100,200

max_depth:
5,10,None

min_samples_leaf:
1,5

Total parameter combinations

3 × 3 × 2 = 18

Using 5-fold cross validation:

18 × 5 = 90 model fits

Best Parameters

Use the values printed by your GridSearchCV output.

Grid Search evaluates every possible parameter combination and generally finds the optimal solution but requires considerable computation.

Randomized Search evaluates only randomly selected combinations, reducing computational time while often finding near-optimal solutions.

## Hyperparameter Tuning with GridSearchCV

Hyperparameter tuning was performed on the Random Forest classifier using `GridSearchCV`. The objective was to systematically identify the combination of Random Forest hyperparameters that achieved the highest cross-validated ROC-AUC score.

A Scikit-learn Pipeline was created using `SimpleImputer(strategy='median')`, `StandardScaler()`, and `RandomForestClassifier(random_state=42)`.

Using a Pipeline ensures that preprocessing operations are performed within each cross-validation fold. The imputer and scaler are fitted only on the training portion of each fold, reducing the risk of data leakage.

### Hyperparameter Search Space

The following Random Forest hyperparameters were evaluated:

* `n_estimators`: 50, 100, 200
* `max_depth`: 5, 10, None
* `min_samples_leaf`: 1, 5

The total number of hyperparameter combinations was:

3 × 3 × 2 = 18 configurations

Each configuration was evaluated using five-fold stratified cross-validation. Therefore, the total number of cross-validation model fits was:

18 × 5 = 90 model fits

### Best Hyperparameters

GridSearchCV identified the following best parameter combination:

* `max_depth = None`
* `min_samples_leaf = 1`
* `n_estimators = 200`

The best mean cross-validated ROC-AUC score was **0.9331**.

The selected `max_depth=None` allows each Decision Tree in the Random Forest to grow without a predefined depth restriction. The value `min_samples_leaf=1` allows leaf nodes to contain a minimum of one training sample.

The selected `n_estimators=200` means that the Random Forest combines predictions from 200 Decision Trees. Increasing the number of trees can improve the stability of the ensemble because predictions are averaged across a larger collection of models.

### Comparison with the Baseline Random Forest

The original Random Forest achieved a mean five-fold cross-validated ROC-AUC of **0.9260**.

After hyperparameter tuning, the best Random Forest configuration achieved a mean cross-validated ROC-AUC of **0.9331**.

The absolute improvement in ROC-AUC was:

0.9331 − 0.9260 = 0.0071

This represents a modest but measurable improvement in the model's ability to distinguish between the two `loan_status` classes.

The result indicates that the original Random Forest hyperparameters were already effective. However, GridSearchCV identified a more suitable configuration by increasing the number of trees from 100 to 200 and allowing unrestricted tree depth.

### Interpretation of the Best Configuration

The selection of `max_depth=None` suggests that deeper trees were beneficial when used as part of the Random Forest ensemble.

Although an unrestricted individual Decision Tree can have a high risk of overfitting, Random Forest reduces this risk by combining multiple trees trained using bootstrap samples and random feature subsets.

The selection of `min_samples_leaf=1` indicates that the Grid Search did not find additional leaf-size regularization beneficial within the tested parameter grid.

The selection of 200 estimators suggests that averaging predictions across a larger number of trees provided stronger cross-validated predictive performance.

### Grid Search vs Randomized Search

Grid Search exhaustively evaluates every hyperparameter combination specified in the parameter grid. This provides a systematic comparison and guarantees that every defined configuration is evaluated.

However, the computational cost of Grid Search increases rapidly as the number of hyperparameters and candidate values grows. A larger parameter grid may require hundreds or thousands of model training operations.

Randomized Search evaluates only a specified number of randomly selected hyperparameter combinations. It is generally more computationally efficient for large search spaces and allows a wider range of parameter values to be explored.

The disadvantage of Randomized Search is that it does not evaluate every possible parameter combination and may therefore miss a strong configuration.

For this project, GridSearchCV was appropriate because the parameter grid contained only 18 configurations. With five-fold cross-validation, 90 model fits were required, making exhaustive search computationally manageable.

### Conclusion

GridSearchCV successfully improved the Random Forest model's mean cross-validated ROC-AUC from **0.9260 to 0.9331**.

The best configuration used **200 trees, unrestricted tree depth, and a minimum of one sample per leaf**.

The ROC-AUC improvement of **0.0071** is relatively modest but demonstrates that systematic hyperparameter tuning can improve model performance beyond manually selected parameters.

Based on the GridSearchCV results, the tuned Random Forest Pipeline is a strong candidate for final model evaluation and production serialization.


------------------------------------------------------------
7. Manual Learning Curve
------------------------------------------------------------
Training fractions

20%
40%
60%
80%
100%

Interpretation

Training AUC remained extremely high throughout.

Test AUC consistently increased as more training data became available.

This indicates that the model continues benefiting from additional training data.

Because the test AUC was still increasing at 100% of the available training data, the model appears to be primarily data-limited rather than capacity-limited.


Manual Learning Curve Analysis

A manual learning curve was created using the best Random Forest Pipeline identified through GridSearchCV. The objective was to evaluate how the model's training and test ROC-AUC change as progressively larger portions of the training dataset are used.

The tuned Random Forest Pipeline was trained using 20%, 40%, 60%, 80%, and 100% of the available training data. For each training fraction, ROC-AUC was calculated on the corresponding training subset and on the fixed test dataset.

### Learning Curve Results

| Training Fraction | Training AUC | Test AUC |
| ----------------- | -----------: | -------: |
| 20%               |       1.0000 |   0.9181 |
| 40%               |       1.0000 |   0.9248 |
| 60%               |       1.0000 |   0.9319 |
| 80%               |       1.0000 |   0.9352 |
| 100%              |       1.0000 |   0.9389 |

### Training AUC Analysis

The training ROC-AUC remained at **1.0000 for every training fraction**. Therefore, the training AUC did not decrease as the training dataset increased.

A training AUC of 1.0 indicates that the Random Forest perfectly ranks the positive and negative classes within the training data. This behaviour is consistent with the high flexibility of the selected model configuration, particularly because GridSearchCV selected `max_depth=None` and `min_samples_leaf=1`.

These hyperparameters allow individual trees to grow deeply and create highly specific decision boundaries. Therefore, the model demonstrates signs of high training-set fit and potential overfitting.

However, perfect training performance alone does not prove that the model fails to generalize. The test ROC-AUC must also be examined.

### Test AUC Analysis

The test ROC-AUC consistently increased as the amount of training data increased.

Test AUC improved from **0.9181 at 20% of the training data** to **0.9248 at 40%** and **0.9319 at 60%**.

Performance continued to improve to **0.9352 at 80%** and reached its highest value of **0.9389 when 100% of the training data was used**.

This consistent upward trend indicates that the Random Forest benefits from additional training examples. As more observations are introduced, the model learns patterns that generalize more effectively to unseen loan applications.

### Is the Model Limited by Data Quantity or Model Capacity?

The test ROC-AUC is still increasing at the 100% training fraction. There is no clear plateau in test performance within the available training-data range.

Therefore, the current model appears to be more limited by **data quantity than model capacity**.

Collecting additional representative credit-risk observations may further improve the model's generalization performance.

Although the training AUC remains at 1.0, the test AUC also improves consistently as more training data is added. This suggests that additional data is helping reduce the generalization gap.

### Training and Test Performance Gap

At 20% of the training data, the difference between training and test AUC was approximately:

1.0000 − 0.9181 = 0.0819

At 100% of the training data, the difference decreased to approximately:

1.0000 − 0.9389 = 0.0611

The reduction in the training-test AUC gap indicates improved generalization as the amount of training data increases.

The model still demonstrates a high training fit, but the decreasing generalization gap suggests that additional data helps reduce the effect of model variance.

### Conclusion

The manual learning curve shows that training ROC-AUC remains at **1.0000 across all training fractions**, indicating that the tuned Random Forest is highly flexible and fits the training data extremely closely.

However, test ROC-AUC consistently increases from **0.9181 to 0.9389** as the training dataset grows from 20% to 100%.

Since test performance is still improving at the maximum available training fraction, the learning curve suggests that the model is currently **limited primarily by data quantity rather than model capacity**.

Therefore, collecting additional representative training data could potentially improve model generalization further. The tuned Random Forest remains a strong candidate for deployment, although its perfect training AUC should continue to be monitored as an indication of high model complexity.

------------------------------------------------------------
8. Model Serialization
------------------------------------------------------------
The best pipeline obtained from GridSearchCV was saved using

joblib.dump(best_pipeline,"best_model.pkl")

The saved model was successfully reloaded using

joblib.load("best_model.pkl")

Predictions on new unseen records confirmed that serialization and deserialization worked correctly without retraining.

## Model Serialization and Reload Testing

The best-performing tuned Random Forest Pipeline identified by GridSearchCV was serialized to disk using the `joblib` library.

The Pipeline contains three stages: median-value imputation using `SimpleImputer`, feature scaling using `StandardScaler`, and the tuned `RandomForestClassifier`.

The tuned Random Forest uses 200 estimators, unrestricted tree depth, and a minimum of one sample per leaf.

The complete Pipeline was saved as `best_model.pkl` using `joblib.dump()`.

### Why the Complete Pipeline Was Serialized

The complete preprocessing and modeling Pipeline was saved instead of saving only the Random Forest classifier.

This ensures that new input data passes through the same median imputation and feature-scaling transformations used during model training before predictions are generated.

Packaging preprocessing and prediction logic together reduces the risk of inconsistencies between training and production environments.

### Model Reload and Prediction Test

The serialized model was reloaded using `joblib.load('best_model.pkl')`.

Two hand-crafted applicant records were created using the same feature structure expected by the trained model. The reloaded Pipeline successfully generated class predictions and prediction probabilities for both test records without errors.

This confirms that the serialized Pipeline can be loaded in a new Python session and used for inference without retraining the model.

### Production Reproducibility

Model serialization improves reproducibility because the exact fitted preprocessing transformations and trained Random Forest parameters are preserved.

The saved `best_model.pkl` file can be loaded by a future application or inference service to generate predictions on new loan applicant data.

The successful reload-and-predict test confirms that the model artifact is suitable for integration into the later LLM-powered financial advisory system.

### Repository Storage

The serialized model file was checked for its total file size. If the file size remains below the repository's 100 MB file limit, `best_model.pkl` can be committed directly to the project repository.

If the model artifact exceeds 100 MB, the model file should not be committed directly. Instead, the training and GridSearchCV script can be included so that the model artifact can be regenerated reproducibly.

### Conclusion

The tuned Random Forest Pipeline was successfully serialized as `best_model.pkl`, reloaded using `joblib`, and tested on two hand-crafted input records.

The reloaded model generated predictions without errors, confirming that the complete preprocessing and classification workflow has been preserved successfully.

This serialized Pipeline represents the final trained machine-learning artifact and can be reused for future loan-risk prediction and financial advisory integration.


------------------------------------------------------------
9. Final Model Comparison
------------------------------------------------------------
Include a Markdown table with

Model
5-Fold Mean ROC-AUC
5-Fold Std ROC-AUC
Test ROC-AUC

Recommendation

Based on cross-validation, robustness and deployment readiness, the tuned Random Forest Pipeline is recommended for production.

Reasons

• highest and most stable ROC-AUC
• low variance across folds
• robust ensemble learning
• automatic preprocessing inside Pipeline
• easily deployable using best_model.pkl

Summary Model Comparison

The classification models developed in Parts 2 and 3 were compared using 5-fold cross-validation ROC-AUC, cross-validation standard deviation, and test-set ROC-AUC.

| Model                    | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
| ------------------------ | -----------------: | ----------------: | -----------: |
| Logistic Regression      |             0.8638 |            0.0029 |       0.8603 |
| Controlled Decision Tree |             0.8913 |            0.0038 |       0.8937 |
| Random Forest            |             0.9260 |            0.0030 |       0.9306 |
| Gradient Boosting        |             0.9280 |            0.0022 |       0.9336 |

Model Performance Interpretation

Logistic Regression achieved a mean cross-validation AUC of **0.8638** and a test AUC of **0.8603**. The similar cross-validation and test results indicate stable generalization, although its class-separation performance was lower than the tree-based ensemble models.

The Controlled Decision Tree improved performance with a mean cross-validation AUC of **0.8913** and a test AUC of **0.8937**. The close agreement between these values indicates that controlling tree depth reduced overfitting compared with the unconstrained Decision Tree.

Random Forest achieved a strong mean cross-validation AUC of **0.9260** and a test AUC of **0.9306**. This demonstrates that ensemble averaging substantially improved class-separation performance compared with a single Decision Tree.

Gradient Boosting produced the strongest overall results among the cross-validated baseline models, with the highest mean cross-validation AUC of **0.9280**, the lowest AUC standard deviation of **0.0022**, and the highest test AUC of **0.9336**.

Client Recommendation

I recommend the **Gradient Boosting model** for the client because it achieved the highest mean cross-validation AUC and test-set AUC among the directly compared models. Its low cross-validation standard deviation indicates stable performance across different validation folds. The model provides stronger class-separation performance than Logistic Regression, the Controlled Decision Tree, and the baseline Random Forest. Therefore, Gradient Boosting provides the best balance of predictive performance, stability, and generalization for the current loan-risk classification task.

------------------------------------------------------------
Conclusion
------------------------------------------------------------
This part demonstrated a complete advanced machine learning workflow including overfitting analysis, ensemble learning, feature importance analysis, feature ablation, cross-validation, hyperparameter optimization, learning curve analysis, pipeline construction and model serialization. The tuned Random Forest Pipeline provides the best balance between predictive performance, robustness and deployment readiness, making it the recommended model for the client.
