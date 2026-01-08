Loan Approval Prediction Model
This project aims to predict loan approval outcomes using various machine learning classifiers. By analyzing applicant data such as credit scores, income, and loan amounts, the model identifies the key drivers of creditworthiness and provides an automated decision-making framework.

Project Overview
The workflow involved extensive Exploratory Data Analysis (EDA), feature engineering (including the creation of a Loan-to-Income ratio), and a comparison of several machine learning models to determine the most effective approach for risk assessment.


Key Findings from EDA
Credit Score: Identified as the primary gatekeeper for loan approval.

DTI Ratio: Engineered a Debt-to-Income (DTI) feature which proved more predictive than raw income or loan amount alone.

Feature Pruning: Removed highly correlated summary "points" to eliminate data leakage and dropped low-impact features like "years employed" to reduce noise.

Model Performance
After addressing data leakage, the models were evaluated on a test set (20% of the data). The Random Forest Classifier emerged as the top performer.

Model,Accuracy,Precision (Approved),Recall (Approved)

Random Forest,94.8%,0.91,0.97

Decision Tree,92.1%,0.88,0.95

Logistic Regression,90.5%,0.90,0.88


