# capstone-ai-ml
capstone-ai-ml


Project Title

Loan Approval Prediction and AI-Powered Financial Advisor using Machine Learning and Large Language Models (LLMs)



Problem Statement

Financial institutions receive thousands of loan applications every day. Manually reviewing each application is time-consuming, prone to human error, and may lead to inconsistent decisions. Banks need an intelligent system that can analyze an applicant's financial information, predict the likelihood of loan approval or default, and provide clear explanations for its decisions.

The objective of this project is to develop an end-to-end AI-powered loan approval system that predicts whether a loan application is likely to be approved or rejected based on customer financial information. In addition to prediction, the system will use a Large Language Model (LLM) to explain the decision in plain language and provide personalized financial advice to applicants.

This project demonstrates the complete AI lifecycle—from data preprocessing and exploratory data analysis to machine learning model development, LLM integration, and production-ready deployment with safety guardrails.



Project Objectives

The project aims to:

* Clean and preprocess raw loan application data.
* Explore customer financial characteristics using data analysis and visualization.
* Build a machine learning model to predict loan approval or loan default.
* Explain predictions using a Large Language Model.
* Provide financial recommendations based on customer profiles.
* Build a production-conscious AI assistant with appropriate validation and safety measures.



About the Dataset

The dataset contains information about loan applicants, including demographic details, financial status, employment history, and loan characteristics.

Typical features include:

| Feature                    | Description                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| person_age                 | Applicant's age                                                                                |
| person_income              | Annual income                                                                                  |
| person_home_ownership      | Home ownership status                                                                          |
| person_emp_length          | Years of employment                                                                            |
| loan_intent                | Purpose of the loan                                                                            |
| loan_grade                 | Loan grade assigned by the lender                                                              |
| loan_amnt                  | Requested loan amount                                                                          |
| loan_int_rate              | Loan interest rate                                                                             |
| loan_percent_income        | Loan amount as a percentage of annual income                                                   |
| cb_person_default_on_file  | Previous default history                                                                       |
| cb_person_cred_hist_length | Credit history length                                                                          |
| loan_status                | **Target variable** (Approved/Rejected or Default/No Default depending on the dataset version) |



# Why this dataset is perfect for your capstone

My assignment has four independent parts. This dataset naturally supports each one.

---

Part 1 – Data Preparation and Exploratory Data Analysis (EDA)

Assignment Requirements

* Load dataset
* Handle null values
* Detect duplicates
* Correct data types
* Descriptive statistics
* Skewness
* Outlier detection
* Visualizations
* Correlation analysis
* Save cleaned dataset

Why this dataset fits

It contains:

* Numerical columns
* Categorical columns
* Continuous variables
* Binary variables
* Missing values (or can be reasonably handled if present)
* Multiple financial variables suitable for visualization

Examples:

Histogram

* person_income
* loan_amnt

Scatter Plot

* person_income vs loan_amnt

Box Plot

* loan_grade vs loan_int_rate

Correlation Heatmap

* All numeric financial variables

Outlier Detection

* Income
* Loan Amount
* Interest Rate

This makes Part 1 meaningful instead of merely fulfilling the assignment mechanically.

---

Part 2 – Machine Learning

The cleaned dataset becomes the input for machine learning.

Target:

```text
loan_status
```

Possible algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost (optional)

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Feature Importance

Examples:

* Income
* Loan Amount
* Credit History
* Employment Length

These provide meaningful insights into the factors influencing loan decisions.

---

Part 3 – LLM Integration

This is where your project becomes more than just a classifier.

Instead of displaying:

```text
Prediction: Loan Rejected
```

the LLM can generate a user-friendly explanation, for example:

> Based on your application, the loan was predicted to be rejected because your annual income is relatively low compared to the requested loan amount, and your credit history indicates previous defaults. Improving your credit score, reducing your requested loan amount, or increasing your down payment could improve your chances of approval.

The LLM converts numerical outputs into explanations that are easy for non-technical users to understand.

---

Part 4 – AI Financial Advisor

This is where your project stands out.

The LLM can also provide personalized financial guidance, for example:

Input:

```text
Age: 28
Income: $45,000
Loan Amount: $30,000
Interest Rate: 15%
Previous Default: Yes
```

Output:

My requested loan amount represents a large proportion of your annual income, which increases lending risk. Consider applying for a smaller loan or improving my repayment history before reapplying. Paying existing debts on time and reducing your debt-to-income ratio can improve future approval chances.

This demonstrates how predictive analytics and generative AI can work together.

---

Production Guardrails

My assignment specifically mentions "production-conscious AI".

Examples of guardrails include:

* Validate numerical inputs (e.g., age cannot be negative).
* Reject unrealistic values (e.g., income cannot be negative).
* Avoid making financial guarantees.
* State that recommendations are informational and not professional financial advice.
* Avoid collecting sensitive personal information unnecessarily.
* Handle missing or incomplete user inputs gracefully.

These practices make the system safer and more reliable.

---

Why this dataset is stronger than your previous one

| Previous Dataset                    | Credit Risk Dataset                                          |
| ----------------------------------- | ------------------------------------------------------------ |
| Synthetic data                      | Based on realistic financial records                         |
| Weak feature relationships          | More meaningful relationships among variables                |
| Correlation values mostly near zero | Features have more predictive value                          |
| Limited feature importance          | Better suited for feature selection and model interpretation |
| Less engaging AI explanations       | Rich financial context for explanations and advice           |

---

Overall Project Flow

```text
Raw Credit Risk Dataset
            │
            ▼
Part 1
Data Cleaning + EDA
            │
            ▼
cleaned_data.csv
            │
            ▼
Part 2
Machine Learning Model
(Loan Status Prediction)
            │
            ▼
Prediction
            │
            ▼
Part 3
LLM Explanation
("Why was this prediction made?")
            │
            ▼
Part 4
AI Financial Advisor
(Personalized recommendations + production guardrails)
```

