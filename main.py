import streamlit as st
import requests

st.set_page_config(page_title="CreditMatch AI", page_icon="💳")

st.title("💳 CreditMatch AI")
st.markdown("### Discover Your Credit Eligibility in Seconds")
st.write("Our advanced AI analyzes your profile via our FastAPI backend.")

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Profile")
    annual_income_input = st.number_input("Annual Income (USD)", min_value=0, value=50000, step=1000)
    employed_years_input = st.number_input("Years of Employment", min_value=0, value=2)
    children_input = st.number_input("Number of Children", min_value=0, value=0)
    
with col2:
    st.subheader("Demographics")
    age_input = st.slider("Your Age", 18, 100, 25)
    education_input = st.selectbox("Highest Education", [
        "Higher education", "Secondary / secondary special", 
        "Incomplete higher", "Lower secondary", "Academic degree"
    ])

st.subheader("Ownership Details")
c3, c4 = st.columns(2)
with c3:
    car_input = st.radio("Do you own a Car?", ["Yes", "No"], horizontal=True)
with c4:
    property_input = st.radio("Do you own Property?", ["Yes", "No"], horizontal=True)

car_value = 1 if car_input == "Yes" else 0
property_value = 1 if property_input == "Yes" else 0

education_mapping = {
    "Academic degree": 0, "Higher education": 1, 
    "Incomplete higher": 2, "Lower secondary": 3,
    "Secondary / secondary special": 4
}
education_val = education_mapping[education_input]

if st.button("Analyze My Eligibility", use_container_width=True):
    payload = {
        "Car_Owner": car_value,
        "Propert_Owner": property_value,
        "CHILDREN": int(children_input),
        "EDUCATION": education_val,
        "Annual_income": float(annual_income_input),
        "age": int(age_input),
        "Employed_years": float(employed_years_input)
    }
    
# features = ["Car_Owner", "Propert_Owner", "CHILDREN", "EDUCATION", "Annual_income", "age", "Employed_years"]

    try:
        # Call the FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            decision = result['decision']
            risk_prob = result['risk_probablity']
            
            st.divider()
            
            if decision == "Approved":
                st.balloons()
                st.success(f"### Great News! You have a strong profile.")
                st.metric(label="Calculated Risk", value=f"{risk_prob:.1%}")
            else:
                st.warning(f"### We're sorry, it's not a match right now.")
                st.write(f"**Reason:** {result['explanation']}")
                st.metric(label="Calculated Risk", value=f"{risk_prob:.1%}")
                
        else:
            st.error(f"API Error: Received status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Did you run: `uvicorn app:app --reload`?")

with st.expander("🔍 Why did the AI make this decision?"):
    st.write("Based on your financial profile and asset ownership:")
    
    if annual_income_input > 70000:
        st.info("✅ High Annual Income is a strong positive factor for your approval.")
    if car_value == 1 and property_value == 1:
        st.info("✅ Asset ownership (Car/Property) significantly reduces your risk profile.")
    if employed_years_input < 2:
        st.warning("⚠️ Short employment history increases the calculated risk score.")
    if children_input > 3:
        st.warning("⚠️ Large household size relative to income can impact eligibility.")