CreditMatch AI: End-to-End Credit Approval System
Project Overview
CreditMatch AI is a machine-learning-powered application that predicts the likelihood of credit card approval based on applicant demographics and financial history. Using a Random Forest Classifier, the system moves beyond simple binary "Yes/No" predictions to provide a nuanced Approval Confidence Score, enabling financial institutions to adjust their risk tolerance dynamically.

 Key Features
Predictive Modeling: Uses a Random Forest ensemble trained on thousands of credit records.

Dynamic Thresholding: A customizable "Strictness" setting that shifts the model's decision boundary.

Bias Correction: Implements class_weight='balanced' and advanced feature engineering to handle imbalanced datasets.

Interactive UI: A Streamlit web application that provides instant feedback and "Risk Scores" to users.

The Tech Stack
Language: Python 3.12

Data Analysis: pandas, numpy

Visualization: seaborn, matplotlib

Machine Learning: scikit-learn

Deployment: Streamlit

Model Persistence: joblib

 Data & Model Insights
The Imbalance Challenge
One of the core challenges of this project was the highly imbalanced nature of credit data (where approved cases significantly outnumber rejections). To prevent the model from becoming a "Yes Man," I utilized:

Feature Importance: Identifying that Annual_income and Employed_days are the primary drivers of creditworthiness.

Probability Thresholding: Instead of the default 50% cutoff, this app offers an "Elite" 90% confidence threshold for stricter approvals.

Performance Metrics
Accurary: ~87% 
