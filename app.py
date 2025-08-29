import streamlit as st
import joblib
import numpy as np

# Load saved SVM model and vectorizer
model = joblib.load('svm_fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Fake Job Detector 💼❌", layout="centered")

st.title("🚨 Fake Job Recruitment Detection")
st.write("Enter the job details below to predict whether it's **Fake** or **Real**.")

# Input field for job description
job_description = st.text_area("📝 Job Description", height=200)

if st.button("Predict"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Transform input text
        input_data = vectorizer.transform([job_description])
        
        # Predict
        prediction = model.predict(input_data)[0]

        # Note: Not all SVM models have predict_proba unless trained with `probability=True`
        try:
            prob = model.predict_proba(input_data)[0][prediction]
            confidence_text = f"(Confidence: {prob:.2f})"
        except:
            confidence_text = "(Confidence score not available)"

        # Show result
        if prediction == 1:
            st.error(f"❌ This job posting is likely **Fake**. {confidence_text}")
        else:
            st.success(f"✅ This job posting is likely **Real**. {confidence_text}")
