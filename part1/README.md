PART 1 – EXPLORATORY DATA ANALYSIS (EDA)

Dataset Description The Credit Risk Dataset was analyzed to understand
applicant demographics, loan characteristics, employment details, and
credit history. The objective of this EDA was to clean the data and
prepare it for machine learning.

1.  Data Loading The dataset was loaded successfully into a pandas
    DataFrame. The first five rows, data types, and dataset dimensions
    were inspected to verify correct loading.

2.  Missing Value Analysis Missing values were calculated for every
    column. Numeric columns with less than 20% missing values were
    imputed using the median because the median is robust to skewed
    distributions and outliers.

3.  Duplicate Detection Duplicate records were identified and removed.
    The number of duplicates removed was reported. Removing duplicates
    did not significantly affect the missing-value percentages.

4.  Data Type Correction Numeric columns stored as object types were
    converted appropriately, and repetitive string columns were
    converted to category dtype to reduce memory usage.

5.  Descriptive Statistics and Skewness Descriptive statistics were
    computed for all numeric columns. The most positively skewed column
    was person_income, meaning the distribution contains extreme
    high-income values. Because the mean is influenced by these
    outliers, the median is a better measure for missing-value
    imputation.

6.  Outlier Detection The IQR method was applied to identify outliers.
    Outliers were retained because they likely represent genuine
    financial observations, and tree-based machine learning models are
    generally robust to them.

7.  Visualizations • Line Plot: Displayed the trend of a numerical
    feature. • Bar Chart: Compared average loan amount across loan
    grades. • Histogram: Confirmed the strong positive skewness of the
    most skewed feature. • Scatter Plot: Showed a positive relationship
    between person_income and loan_amnt. • Box Plot: Showed differences
    in median and spread of loan amount across loan grades.

8.  Correlation Heatmap Pearson correlation analysis identified
    person_age and cb_person_cred_hist_length as the strongest
    correlated pair. This relationship is likely explained by age and
    accumulated credit history rather than direct causation.

9a. Imputation Strategy Mean and median were compared for the most
skewed columns. Median imputation was selected because the distributions
were positively skewed. After imputation, no missing values remained.

9b. Spearman Correlation Spearman and Pearson correlations were
compared. The largest differences indicated monotonic but non-linear
relationships. Spearman correlation will be considered alongside Pearson
during feature selection.

9c. Grouped Aggregation Categorical Feature: loan_grade Numeric Feature:
loan_amnt

Highest Mean Group: G (17195.70) Highest Standard Deviation Group: F
(8280.34) Highest-to-Lowest Mean Ratio: 2.01

The high within-group standard deviation indicates that loan grade alone
cannot fully predict loan amount because substantial variability exists
within each grade. However, the mean ratio of 2.01 suggests loan grade
contains useful predictive information.

10. Clean Dataset The cleaned dataset was saved as cleaned_data.csv for
    use in Parts 2 and 3.

Conclusion The EDA successfully cleaned and prepared the dataset by
handling missing values, removing duplicates, correcting data types,
examining skewness and outliers, generating visualizations, analyzing
correlations, and producing a clean dataset suitable for machine
learning.

