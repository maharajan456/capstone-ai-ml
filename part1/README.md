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
    
Null Value Analysis

Missing values were identified using df.isnull().sum(), and the percentage of missing values in each column was calculated using:

(df.isnull().sum() / df.shape[0]) * 100

No columns exceeded a 20% missing-value rate, so all columns were retained for further analysis.

Missing values in numeric columns were imputed using the median rather than the mean. The median was chosen because it is more robust to outliers and skewed distributions. Financial variables such as applicant income and loan amount often contain extreme values that can significantly influence the mean, whereas the median better represents the central tendency of the data.

5.  Data Type Correction Numeric columns stored as object types were
    converted appropriately, and repetitive string columns were
    converted to category dtype to reduce memory usage.

Data Type Correction

The dataset's numeric columns were inspected and found to have appropriate numeric data types (int64 and float64). No numeric columns stored as object were identified, so no numeric type correction was required.

The repetitive string columns (person_home_ownership, loan_intent, loan_grade, and cb_person_default_on_file) were converted from object to category using astype('category'). This reduces memory usage because categorical data stores repeated values more efficiently than plain strings.

Memory usage was measured before and after the conversion using df.memory_usage(deep=True).sum(), confirming the optimization.

7.  Descriptive Statistics and Skewness Descriptive statistics were
    computed for all numeric columns. The most positively skewed column
    was person_income, meaning the distribution contains extreme
    high-income values. Because the mean is influenced by these
    outliers, the median is a better measure for missing-value
    imputation.

Descriptive Statistics and Skewness

Descriptive statistics were generated using df.describe() to summarize the numerical variables. The summary includes measures such as count, mean, standard deviation, minimum, maximum, and quartiles, providing an overview of the data distribution.

Skewness was calculated for each numeric column using df[col].skew(). The column with the highest absolute skewness was <Column_Name> with a skewness value of <value>.

If the skewness is positive:

A positive skew indicates that the distribution has a long tail toward higher values. This usually means that a few large observations increase the mean. In such cases, the median is a better measure of central tendency than the mean because it is less affected by extreme values. 
Therefore, the median is preferred for imputing missing values in positively skewed variables.

If the skewness is negative:

A negative skew indicates that the distribution has a long tail toward lower values. A few unusually small observations pull the mean downward. In this situation, the median is again a more representative measure of central tendency and is preferred for missing-value imputation because it is more robust to extreme values.

9.  Outlier Detection The IQR method was applied to identify outliers.
    Outliers were retained because they likely represent genuine
    financial observations, and tree-based machine learning models are
    generally robust to them.
Outlier Detection Using the IQR Method

Outliers were detected using the Interquartile Range (IQR) method. For each selected numeric column, the first quartile (Q1), third quartile (Q3), and IQR (Q3 − Q1) were calculated. Observations below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR were identified as potential outliers.

Interpretation

1. person_income

The analysis identified (replace with your output) outliers in the person_income column. These represent applicants with exceptionally high incomes. Since high-income individuals are genuine customer profiles and not necessarily data errors, these values will be retained during Part 1.

2. loan_amnt

The loan_amnt column contained (replace with your output) outliers. These correspond to unusually large loan requests. Such observations are expected in real-world lending data and therefore should not be removed solely based on statistical thresholds.

Outlier Handling Strategy

Outliers were not removed during data preparation because they may represent valid customer behavior rather than incorrect data. In Part 2, the effect of these outliers on model performance will be evaluated. If they negatively impact model accuracy, techniques such as IQR-based capping (winsorization) or RobustScaler may be applied. Otherwise, the outliers will be retained to preserve valuable information.

10.  Visualizations • Line Plot: Displayed the trend of a numerical
    feature. • Bar Chart: Compared average loan amount across loan
    grades. • Histogram: Confirmed the strong positive skewness of the
    most skewed feature. • Scatter Plot: Showed a positive relationship
    between person_income and loan_amnt. • Box Plot: Showed differences
    in median and spread of loan amount across loan grades.

11.  Correlation Heatmap Pearson correlation analysis identified
    person_age and cb_person_cred_hist_length as the strongest
    correlated pair. This relationship is likely explained by age and
    accumulated credit history rather than direct causation.

Correlation Heat Map

A Pearson correlation matrix was computed for all numeric variables and visualized using a heat map. The strongest positive correlation was observed between person_age and cb_person_cred_hist_length, with a correlation coefficient of 0.8592.

This indicates a strong positive linear relationship, meaning that as a person's age increases, their credit history length also tends to increase.

Does this indicate causation?

Although the correlation is very strong, correlation does not necessarily imply causation.

In this case, however, there is a logical real-world relationship between the two variables.

Older individuals have simply had more time to build a longer credit history, making a strong positive correlation expected.

Therefore, while age itself does not directly "cause" a longer credit history, the passage of time naturally contributes to both variables increasing together.

9a. Imputation Strategy Mean and median were compared for the most
skewed columns. Median imputation was selected because the distributions
were positively skewed. After imputation, no missing values remained.

Imputation Strategy Comparison

The skewness of all numeric columns was computed, and the two columns with the highest absolute skewness were selected for comparison. For each column, both the mean and median were calculated before imputation.

Since these columns exhibited positive skewness, the median was chosen to impute missing values. In positively skewed distributions, a small number of very large values pull the mean upward, making it less representative of the typical observation. The median is resistant to extreme values and therefore provides a better measure of central tendency.

Missing values in the selected columns were filled using fillna(df[col].median()). After imputation, isnull().sum() confirmed that no missing values remained in these columns.

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

