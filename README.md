# customer-churn-prediction-ml-pipeline
End-to-end machine learning pipeline for predicting retail customer churn and identifying high-risk customers using Logistic Regression.
# Customer Churn Prediction ML Pipeline

An end-to-end machine learning pipeline for predicting **customer churn in a retail business** and identifying customers who require proactive retention action.

The project uses **Logistic Regression** to estimate churn risk from customer behaviour indicators such as age, monthly spending, and complaint frequency. The workflow covers the complete journey from raw, messy customer data to a business-ready churn risk report.

---

## Business Problem

Customer churn directly affects revenue, customer lifetime value, and business growth.

SmartKart wants to identify customers who are likely to discontinue their purchases so that the retention team can intervene before the customer leaves.

### Business Question

> **Which customers have a higher probability of churning, and what customer behaviours are associated with increased churn risk?**

The objective is not only to build a predictive model, but also to convert its predictions into an actionable customer-retention list.

---

## Project Objective

This project demonstrates an end-to-end supervised machine learning workflow:

* Clean and validate customer data
* Handle missing and invalid values
* Remove duplicate records
* Detect and treat outliers
* Select relevant business features
* Define the churn target
* Split data into training and testing sets
* Standardise numerical features
* Train a Logistic Regression classifier
* Predict churn and churn probability
* Evaluate model performance
* Interpret feature impact
* Generate a ranked customer churn-risk report

---

## Dataset

The project uses a deliberately messy retail customer dataset containing **100 customer records**.

### Features

| Feature         | Description                                  |
| --------------- | -------------------------------------------- |
| `Customer_ID`   | Unique customer identifier                   |
| `Age`           | Customer age                                 |
| `Monthly_Spend` | Customer's monthly spending                  |
| `Complaints`    | Number of customer complaints                |
| `Churn`         | Target variable: `1` = churn, `0` = no churn |

The dataset intentionally contains real-world-style data quality issues including duplicates, missing values, invalid entries, and extreme outliers.

---

## Machine Learning Pipeline

```text
Raw Customer Data
       ↓
Data Inspection
       ↓
Data Cleaning
       ↓
Outlier Detection & Treatment
       ↓
Feature Selection
       ↓
Target Definition
       ↓
Train-Test Split
       ↓
Feature Standardisation
       ↓
Logistic Regression
       ↓
Churn Prediction
       ↓
Model Evaluation
       ↓
Model Interpretation
       ↓
Customer Risk Ranking
```

The notebook implements this as a 15-step pipeline from data collection through final business output.

---

## Data Preprocessing

The raw dataset contains several data-quality problems that are handled before modelling.

### Cleaning

* Duplicate customer records are removed.
* Unnecessary whitespace is stripped.
* Text-based age values are converted to numeric values.
* Invalid age values are treated as missing.
* Negative monthly spending values are treated as invalid.
* Missing values are replaced using the median.

Median imputation is used because it is less affected by extreme values than mean imputation.

### Outlier Treatment

Extreme values in `Monthly_Spend` and `Complaints` are detected using the **Interquartile Range (IQR)** method.

Instead of deleting affected customer records, outliers are capped using winsorisation so that useful customer information is retained.

---

## Features Used

The model uses three business-relevant predictors:

```text
Age
Monthly_Spend
Complaints
```

`Customer_ID` is excluded from model training because it is an identifier rather than a meaningful predictive feature.

---

## Model

### Logistic Regression

Logistic Regression was selected because churn is a **binary classification problem**.

The model predicts:

```text
0 → No Churn
1 → Churn
```

It also produces a **churn probability**, which is particularly useful for ranking customers according to retention risk.

---

## Model Evaluation

The model is evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The notebook's documented expected evaluation is approximately **89–95% accuracy**, with **very high recall for churn detection** on the test set. Exact metrics should be taken from the executed notebook output rather than assumed from the documentation.

For a customer-retention use case, recall is particularly important because failing to identify an actual churner can result in the loss of a customer.

---

## Key Business Insights

The model coefficients are analysed to understand the relationship between customer behaviour and churn risk.

The notebook identifies:

### 1. Complaints

A higher number of complaints is associated with **higher churn risk**.

This suggests that improving complaint resolution and customer support could be an important retention lever.

### 2. Monthly Spending

Higher monthly spending is associated with **lower churn risk** in this dataset.

High-value customers therefore deserve particular attention because protecting these relationships can have a significant business impact.

### 3. Age

Age has a comparatively smaller relationship with churn than the other two features in this dataset.

These relationships are specific to the project dataset and should not automatically be interpreted as universal customer-behaviour rules.

---

## Business Output

Instead of stopping at model metrics, the project creates a business-ready churn-risk report containing:

```text
Customer_ID
Age
Monthly_Spend
Complaints
Actual_Churn
Predicted_Churn
Churn_Probability
Risk_Label
```

Customers are ranked by churn probability so that the retention team can prioritise the highest-risk customers first.

Example risk labels:

```text
Likely to Churn
Not Likely to Churn
```

The final report is exported as:

```text
smartkart_churn_risk_report.csv
```

---

## Tech Stack

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical operations
* **Matplotlib** — visualisation
* **Seaborn** — confusion matrix visualisation
* **Scikit-learn** — preprocessing, modelling and evaluation
* **Google Colab** — development environment

---

## Project Files

```text
data/
    SmartKart_dirty_100_rows.csv

notebooks/
    SmartKart_Churn_Prediction_ML_Pipeline.ipynb

outputs/
    smartkart_churn_risk_report.csv
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction-ml-pipeline.git
cd customer-churn-prediction-ml-pipeline
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the notebook

Open:

```text
notebooks/SmartKart_Churn_Prediction_ML_Pipeline.ipynb
```

The original notebook is designed for **Google Colab** and includes a file-upload step for the dataset.

### 4. Run all cells

Execute the notebook from top to bottom to reproduce the preprocessing, model training, evaluation and churn-risk report.

---

## Model Workflow

The implementation follows these stages:

**Data Collection → Data Understanding → Data Cleaning → Outlier Treatment → Feature Selection → Target Definition → Target Encoding Verification → Train-Test Split → Feature Standardisation → Model Building → Model Training → Prediction → Evaluation → Interpretation → Final Business Output**

---

## Why This Project Matters

This project demonstrates more than simply training a machine learning model.

It connects:

**Data Quality → Machine Learning → Model Interpretation → Business Decision-Making**

The final objective is to help a retail retention team answer:

> **Who is likely to leave, how likely are they to leave, and which customers should we prioritise for retention?**

---

## Future Improvements

Potential production-level improvements include:

* Add larger real-world customer datasets
* Introduce additional behavioural features such as purchase frequency, recency and tenure
* Compare Logistic Regression with Random Forest, XGBoost and other classification models
* Use cross-validation and hyperparameter tuning
* Add ROC-AUC and Precision-Recall curves
* Handle class imbalance when present
* Build an interactive churn-risk dashboard
* Deploy the model as an API
* Add automated data validation
* Create a production-ready ML pipeline
* Monitor model performance and data drift

---

## Disclaimer

This is a portfolio/academic machine learning project based on a small, intentionally messy retail dataset. Model performance and feature relationships should not be treated as production conclusions without validation on a larger and representative customer dataset.

---

## Author

**Chitransh**

BBA — AI/ML

### Project Focus

**Machine Learning • Customer Analytics • Predictive Modelling • Retail Analytics • Customer Retention**
