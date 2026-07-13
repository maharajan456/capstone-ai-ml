import streamlit as st
import pandas as pd
import joblib
from guardrails import has_pii
from guardrails import call_llm
from guardrails import validate_json
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)
st.markdown("""
<style>

.main{
    padding-top:2rem;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

div[data-testid="stMetric"]{
    border:1px solid #dcdcdc;
    border-radius:12px;
    padding:15px;
    background-color:#fafafa;
}

.stAlert{
    border-radius:10px;
}

h1,h2,h3{
    color:#003366;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
This application predicts whether a loan applicant is likely to default using a trained
**Random Forest Classifier**.

This is Part 4 of the Machine Learning Capstone Project.
""")

# ============================================================
# MODEL STATUS
# ============================================================

if model_loaded:
    st.success("✅ Model loaded successfully.")
else:
    st.error(f"❌ Failed to load model.\n\n{error_message}")

# ============================================================
# SIDEBAR INPUTS
# ============================================================
# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=80
)

st.sidebar.title("Loan Risk Assessment")

st.sidebar.markdown(
    "Fill in the applicant information and click **Predict Loan Risk**."
)

st.sidebar.divider()


st.sidebar.header("Applicant Information")

person_age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

person_income = st.sidebar.number_input(
    "Annual Income",
    min_value=1000,
    value=50000
)

person_emp_length = st.sidebar.number_input(
    "Employment Length (Years)",
    min_value=0,
    max_value=50,
    value=5
)

person_home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    [
        "OWN",
        "RENT",
        "MORTGAGE",
        "OTHER"
    ]
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    [
        "EDUCATION",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.sidebar.selectbox(
    "Loan Grade",
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G"
    ]
)

loan_amnt = st.sidebar.number_input(
    "Loan Amount",
    min_value=500,
    value=10000
)

loan_int_rate = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=40.0,
    value=10.5
)

loan_percent_income = st.sidebar.number_input(
    "Loan Percent of Income",
    min_value=0.00,
    max_value=1.00,
    value=0.25
)

cb_person_default_on_file = st.sidebar.selectbox(
    "Previous Default",
    [
        "N",
        "Y"
    ]
)

cb_person_cred_hist_length = st.sidebar.number_input(
    "Credit History Length",
    min_value=1,
    max_value=40,
    value=5
)

predict_button = st.sidebar.button(
    "🔍 Predict Loan Risk",
    use_container_width=True
)

# ============================================================
# MAIN PAGE
# ============================================================

st.header("Applicant Summary")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Personal Information")

    st.write(f"**Age:** {person_age}")
    st.write(f"**Annual Income:** ₹ {person_income:,}")
    st.write(f"**Employment Length:** {person_emp_length} years")
    st.write(f"**Home Ownership:** {person_home_ownership}")

with col2:

    st.subheader("Loan Information")

    st.write(f"**Loan Purpose:** {loan_intent}")
    st.write(f"**Loan Grade:** {loan_grade}")
    st.write(f"**Loan Amount:** ₹ {loan_amnt:,}")
    st.write(f"**Interest Rate:** {loan_int_rate}%")
    st.write(f"**Loan Percent Income:** {loan_percent_income}")
    st.write(f"**Previous Default:** {cb_person_default_on_file}")
    st.write(f"**Credit History Length:** {cb_person_cred_hist_length} years")

# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_record():

    encoded = {

        # Numerical Features
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_grade": ord(loan_grade) - ord("A"),
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,

        # Home Ownership
        "person_home_ownership_OTHER":
            1 if person_home_ownership == "OTHER" else 0,

        "person_home_ownership_OWN":
            1 if person_home_ownership == "OWN" else 0,

        "person_home_ownership_RENT":
            1 if person_home_ownership == "RENT" else 0,

        # Loan Intent
        "loan_intent_EDUCATION":
            1 if loan_intent == "EDUCATION" else 0,

        "loan_intent_HOMEIMPROVEMENT":
            1 if loan_intent == "HOMEIMPROVEMENT" else 0,

        "loan_intent_MEDICAL":
            1 if loan_intent == "MEDICAL" else 0,

        "loan_intent_PERSONAL":
            1 if loan_intent == "PERSONAL" else 0,

        "loan_intent_VENTURE":
            1 if loan_intent == "VENTURE" else 0,

        # Previous Default
        "cb_person_default_on_file_Y":
            1 if cb_person_default_on_file == "Y" else 0

    }

    return encoded
# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    encoded_record = encode_record()

    input_df = pd.DataFrame([encoded_record])

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.header("Prediction Result")
    
    col1, col2, col3 = st.columns(3)

    prediction_text = "Default" if prediction == 1 else "No Default"

    with col1:
        st.metric(
            "Prediction",
            prediction_text
        )

    with col2:
        st.metric(
            "Default Probability",
            f"{probability*100:.2f}%"
        )

    with col3:
        if prediction == 0:
            st.success("Approved")
        else:
            st.error("Review Required")

    if prediction == 0:
        st.success("✅ Applicant is NOT likely to default.")
    else:
        st.error("⚠ Applicant is likely to default.")

    st.metric(
        label="Probability of Default",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))
    
    st.subheader("Risk Assessment")

    if probability < 0.30:

        st.success("🟢 Low Risk Applicant")

    elif probability < 0.60:

        st.warning("🟡 Medium Risk Applicant")

    else:

        st.error("🔴 High Risk Applicant")

    st.subheader("Prediction Details")
    
    prediction_text = "Default" if prediction == 1 else "No Default"

    result = pd.DataFrame({

        "Prediction":[prediction_text],
        "Predicted Class":[prediction],
        "Default Probability":[f"{probability*100:.2f}%"]

    })

    st.table(result)

    with st.expander("View Encoded Features"):

        st.dataframe(
            input_df,
            use_container_width=True
        )

    # ========================================================
    # BUILD LLM PROMPT
    # ========================================================

    user_prompt = f"""
Applicant Details

Age : {person_age}

Income : {person_income}

Employment Length : {person_emp_length}

Home Ownership : {person_home_ownership}

Loan Purpose : {loan_intent}

Loan Grade : {loan_grade}

Loan Amount : {loan_amnt}

Interest Rate : {loan_int_rate}

Loan Percent Income : {loan_percent_income}

Credit History Length : {cb_person_cred_hist_length}

Previous Default : {cb_person_default_on_file}

Predicted Class : {prediction_text}

Predicted Probability : {probability:.4f}

Explain why the model made this prediction.

Return ONLY valid JSON.
"""

    st.divider()
    st.header("LLM Prediction Explanation")

    # ========================================================
    # PII GUARDRAIL
    # ========================================================

    if has_pii(user_prompt):

        st.error("🚫 Input blocked: PII detected.")

    else:

        st.success("✅ Guardrail Passed (No PII Detected)")

        # ====================================================
        # CALL LLM
        # ====================================================

        with st.spinner("Generating explanation..."):

            response = call_llm(user_prompt)

        if response is None:

            st.error("Failed to receive a response from the LLM.")

        else:

            st.subheader("Raw LLM Response")

            st.code(response, language="json")

            # ================================================
            # JSON VALIDATION
            # ================================================

            explanation, passed = validate_json(response)

            if passed:

                st.success("✅ JSON Validation Passed")

            else:

                st.error("❌ JSON Validation Failed")

            # ================================================
            # DISPLAY EXPLANATION
            # ================================================

            st.subheader("Structured AI Explanation")

            st.json(explanation)
            
            import json

            json_output = json.dumps(
            explanation,
            indent=4
            )

            st.download_button(

            "📥 Download Explanation",

            data=json_output,

            file_name="loan_prediction_explanation.json",

            mime="application/json"

            )

            st.subheader("Copy JSON")

            st.code(
            json_output,
            language="json"
            )

# ============================================================
# ABOUT PROJECT
# ============================================================

            st.divider()

            with st.expander("📖 About This Project"):

                st.markdown("""

### Loan Approval Prediction System

This application predicts the likelihood that a customer will default on a loan using a Machine Learning model trained on a Credit Risk dataset.

### Features

- Random Forest Prediction
- Probability Estimation
- LLM Explanation
- JSON Validation
- PII Guardrails
- Streamlit Interface

### Technologies

- Python
- Streamlit
- Scikit-Learn
- OpenRouter API
- JSON Schema

""")
            st.divider()

            st.markdown(
"""
<center>

### 👨‍💻 Developed by Maharajan 

Machine Learning Capstone Project

2026

</center>
""",
unsafe_allow_html=True
)