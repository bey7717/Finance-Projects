import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load(r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Credit Card Approval Model\credit_model.pkl')
scaler = joblib.load(r'C:\Users\bey77\OneDrive\Desktop\Projects\FinanceProj\Credit Card Approval Model\scaler.pkl')

st.set_page_config(page_title="CreditMatch AI", page_icon="💳")

st.title("💳 CreditMatch AI")
st.markdown("### Discover Your Credit Eligibility in Seconds")
st.write("Our advanced AI analyzes your profile to provide instant feedback on your credit card application.")

threshold = 0.10
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Profile")
    annual_income_input = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
    employed_years_input = st.number_input("Years of Employment", min_value=0, value=2)
    employed_days = employed_years_input * -365
    
with col2:
    st.subheader("Demographics")
    age_input = st.slider("Your Age", 18, 100, 25)
    education_input = st.selectbox("Highest Education", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])

st.subheader("Ownership Details")
c3, c4 = st.columns(2)
with c3:
    car_input = st.radio("Do you own a Car?", ["Yes", "No"], horizontal=True)
with c4:
    property_input = st.radio("Do you own Property?", ["Yes", "No"], horizontal=True)

# Processing
car_value = 1 if car_input == "Yes" else 0
property_value = 1 if property_input == "Yes" else 0

education_mapping = {
    "Academic degree": 0, "Higher education": 1, 
    "Incomplete higher": 2, "Lower secondary": 3,
    "Secondary / secondary special": 4
}
education_val = education_mapping[education_input]

input_data = np.array([[car_value, property_value, annual_income_input, education_val, employed_days, age_input]])

if st.button("Analyze My Eligibility", use_container_width=True):
    scaled_input = scaler.transform(input_data)
    model_prob = model.predict_proba(scaled_input)
    
    acceptance_prob = model_prob[0][0]
    rejection_prob = model_prob[0][1]

    st.divider()
    
    res_col1, res_col2 = st.columns(2)
    
    # with res_col1:
    #     st.metric(label="Approval Confidence", value=f"{acceptance_prob:.1%}")
    # with res_col2:
    #     st.metric(label="Calculated Risk", value=f"{rejection_prob:.1%}")

    # st.write("**Eligibility Gauge**")
    # st.progress(acceptance_prob)

    if rejection_prob < threshold:
        st.balloons()
        st.success(f"### Great News! You have a strong profile.")
        st.write("Based on our AI analysis, your application has a high probability of approval.")
        with res_col1:
            st.metric(label="Approval Confidence", value=f"{acceptance_prob:.1%}")
        with res_col2:
            st.metric(label="Calculated Risk", value=f"{rejection_prob:.1%}")
    else:
        st.warning(f"### We're sorry, it's not a match right now.")
        st.write("Our AI indicates you don't quite meet the threshold for this specific card. **Don't give up!** Increasing your length of employment or reporting additional income could improve your score in the future.")
