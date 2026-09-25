import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>

        .title {
            text-align: center;
            font-size: 38px;
            color: #6A4C93;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #4B4B4B;
            margin-bottom: 30px;
        }

        .stButton>button {
            width: 100%;
            background-color: #DCC6F0;
            color: #4A2C6A;
            font-size: 18px;
            font-weight: 600;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #C5A8E0;
        }

        .stButton>button:hover {
            background-color: #C5A8E0;
            color: #3B2055;
            border: 1px solid #A985C7;
        }

        .footer {
            text-align: center;
            color: #888888;
            font-size: 14px;
            margin-top: 30px;
        }

    </style>
""", unsafe_allow_html=True)

model=joblib.load("loan_eligibility_model.pkl")
scaler=joblib.load("scaler.pkl")

st.markdown(
    '<div class="title"> Loan Eligibility Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Smart Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

col1, col2=st.columns(2)

with col1:
    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married=st.selectbox(
        "Married",
        ["Yes","No"]
    )

    dependents=st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education=st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    self_employed=st.selectbox(
        "Self Employed",
        ["Yes","No"]
    )

    applicantincome=st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

with col2:
    coapplicantincome=st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    loanamount=st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100.0
    )

    loan_amount_term=st.number_input(
        "Loan Amount Term",
        min_value=0.0,
        value=360.0
    )

    credit_history=st.selectbox(
        "Credit History",
        [1.0,0.0]
    )

    property_area=st.selectbox(
        "Property Area",
        ["Urban","Semiurban","Rural"]
    )

st.write("")

predict_btn=st.button(" Predict Loan Eligibility")

if predict_btn:
    gender=1 if gender=="Male" else 0

    married=1 if married=="Yes" else 0

    dependents={
        "0":0,
        "1":1,
        "2":2,
        "3+":3
    }[dependents]

    education=1 if education=="Graduate" else 0

    self_employed=1 if self_employed=="Yes" else 0

    property_area={
        "Urban":2,
        "Semiurban":1,
        "Rural":0
    }[property_area]

    input_data=np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicantincome,
        coapplicantincome,
        loanamount,
        loan_amount_term,
        credit_history,
        property_area
    ]])

    input_data=scaler.transform(input_data)

    pred=model.predict(input_data)[0]

    prob=model.predict_proba(input_data)[0][1]

    if pred==1:
        st.success(
            f"✅ Loan Eligible\n\n"
            f"**Probability of Approval:** {prob:.2%}"
        )
    else:
        st.error(
            f"❌ Loan Not Eligible\n\n"
            f"**Probability of Approval:** {prob:.2%}"
        )

st.write("---")

st.markdown(
    "<p class='footer'>© 2026 Loan Eligibility Predictor - All Rights Reserved.</p>",
    unsafe_allow_html=True
)
