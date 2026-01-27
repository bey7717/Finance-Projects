# Credit Card Approval Model

An AI-powered credit card approval system using Random Forest classification with a Streamlit frontend and FastAPI backend.

## Features
- ML model trained on credit card application data
- Streamlit web interface for real-time predictions
- FastAPI REST API backend
- dbt data transformation pipeline
- SHAP explainability analysis

## Setup
```bash
pip install -r requirements.txt
```

## Running the Application
1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

2. In another terminal, run Streamlit:
   ```bash
   streamlit run main.py
   ```

## Project Structure
- `app.py` - FastAPI backend with prediction endpoint
- `main.py` - Streamlit frontend
- `analysis.ipynb` - Model training and evaluation
- `connect_db.py` - Database connection utilities
- `my_dbt_project/` - dbt models for data transformation
- `requirements.txt` - Python dependencies

## Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 7 input features (Car Owner, Property Owner, Children, Education, Annual Income, Age, Employment Years)
- **Training Data**: Credit card application dataset
- **Decision Threshold**: 0.20 risk probability

## License
MIT
