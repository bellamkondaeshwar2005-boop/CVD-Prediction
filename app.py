"""Educational heart disease risk demo built with Streamlit and scikit-learn."""

import streamlit as st

from heart_model import get_model


st.set_page_config(page_title="Heart Health Risk Demo", page_icon="🫀", layout="centered")
st.title("🫀 Heart Health Risk Demo")
st.caption("An educational machine-learning demonstration using the UCI Cleveland dataset.")
st.warning(
    "For education only. This app is not a medical device and cannot diagnose or rule out heart disease. "
    "Do not use its output to make health decisions; speak with a qualified clinician."
)

with st.expander("About this demo"):
    st.write(
        "A logistic-regression pipeline is trained from the Cleveland subset of the UCI Heart Disease dataset. "
        "The original outcome has multiple severity values; this demo groups any positive value as disease present. "
        "The displayed score is a model probability on a small historical dataset, not an individual's clinical risk."
    )

with st.form("patient_inputs"):
    st.subheader("Enter the dataset features")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55)
        sex = st.selectbox("Sex (dataset coding)", [1, 0], format_func=lambda x: "Male (1)" if x else "Female (0)")
        cp = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda x: ["Typical angina (0)", "Atypical angina (1)", "Non-anginal pain (2)", "Asymptomatic (3)"][x])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=70, max_value=250, value=130)
        chol = st.number_input("Serum cholesterol (mg/dL)", min_value=80, max_value=700, value=240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
    with right:
        thalach = st.number_input("Maximum heart rate achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
        ca = st.selectbox("Major vessels colored by fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia code", [3, 6, 7], format_func=lambda x: {3: "Normal (3)", 6: "Fixed defect (6)", 7: "Reversible defect (7)"}[x])
    submitted = st.form_submit_button("Run educational demo")

if submitted:
    values = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    try:
        model = get_model()
        probability = float(model.predict_proba(values)[0, 1])
        st.metric("Model score for the positive class", f"{probability:.0%}")
        st.info("This score is not a diagnosis or a validated personal risk estimate. Please consult a clinician for interpretation.")
    except Exception as exc:
        st.error("The model could not be prepared. Check your internet connection and try again.")
        with st.expander("Technical details"):
            st.code(str(exc))
