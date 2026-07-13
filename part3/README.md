
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

------------------------------------------------------------
8. Model Serialization
------------------------------------------------------------
The best pipeline obtained from GridSearchCV was saved using

joblib.dump(best_pipeline,"best_model.pkl")

The saved model was successfully reloaded using

joblib.load("best_model.pkl")

Predictions on new unseen records confirmed that serialization and deserialization worked correctly without retraining.

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

------------------------------------------------------------
Conclusion
------------------------------------------------------------
This part demonstrated a complete advanced machine learning workflow including overfitting analysis, ensemble learning, feature importance analysis, feature ablation, cross-validation, hyperparameter optimization, learning curve analysis, pipeline construction and model serialization. The tuned Random Forest Pipeline provides the best balance between predictive performance, robustness and deployment readiness, making it the recommended model for the client.
