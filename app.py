import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('spam_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

st.markdown("""
# 📧 Email Spam Classifier  
Enter your email's numeric features and check if it's **Spam** or **Not Spam**!
---
""")

# Sidebar for inputs
st.sidebar.header("Enter Email Feature Values")

feature_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
    'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total'
]

# Create input sliders dynamically
input_features = []
for feature in feature_names:
    value = st.sidebar.slider(f"{feature}", min_value=0.0, max_value=10.0, step=0.01)
    input_features.append(value)

# Predict button
if st.sidebar.button("🔍 Predict Spam or Not"):
    input_data = np.array([input_features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("🚫 This email is classified as: **SPAM**!")
    else:
        st.success("✅ This email is classified as: **NOT SPAM**!")

st.sidebar.markdown("---")
st.sidebar.info("Adjust values and click Predict to classify your email.")

