# app.py - Streamlit Web Application
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ---- Page Config ----
st.set_page_config(
    page_title='Placement Predictor',
    page_icon='🎓',
    layout='wide'
)

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = joblib.load('models/placement_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

# ---- Title ----
st.title('🎓 Placement Prediction System')
st.markdown('*Enter your details to check your placement chances!*')
st.markdown('---')

# ---- Input Form ----
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Academic Details')
    ssc_p = st.slider('10th Percentage (SSC)', 40.0, 100.0, 70.0, 0.5)
    hsc_p = st.slider('12th Percentage (HSC)', 40.0, 100.0, 70.0, 0.5)
    degree_p = st.slider('Degree Percentage', 40.0, 100.0, 65.0, 0.5)
    mba_p = st.slider('MBA Percentage', 40.0, 100.0, 60.0, 0.5)

with col2:
    st.subheader('Background')
    gender = st.selectbox('Gender', ['M', 'F'])
    ssc_b = st.selectbox('10th Board', ['Central', 'Others'])
    hsc_b = st.selectbox('12th Board', ['Central', 'Others'])
    hsc_s = st.selectbox('12th Stream', ['Science', 'Commerce', 'Arts'])
    degree_t = st.selectbox('Degree Type', ['Sci&Tech', 'Comm&Mgmt', 'Others'])

with col3:
    st.subheader('Experience')
    workex = st.selectbox('Work Experience', ['Yes', 'No'])
    etest_p = st.slider('E-test Percentage', 40.0, 100.0, 65.0, 0.5)
    specialisation = st.selectbox('MBA Specialisation', ['Mkt&HR', 'Mkt&Fin'])

st.markdown('---')

# ---- Predict Button ----
if st.button('🔮 Predict My Placement Chances', use_container_width=True):

    # Encode inputs same as training
    gender_enc = 1 if gender == 'M' else 0
    ssc_b_enc = 0 if ssc_b == 'Central' else 1
    hsc_b_enc = 0 if hsc_b == 'Central' else 1
    hsc_s_enc = {'Science': 2, 'Commerce': 0, 'Arts': 1}.get(hsc_s, 0)
    degree_t_enc = {'Sci&Tech': 2, 'Comm&Mgmt': 0, 'Others': 1}.get(degree_t, 0)
    workex_enc = 1 if workex == 'Yes' else 0
    spec_enc = 0 if specialisation == 'Mkt&Fin' else 1

    # Build input array - order must match training features
    input_data = pd.DataFrame([[
        gender_enc, ssc_p, ssc_b_enc, hsc_p, hsc_b_enc,
        hsc_s_enc, degree_p, degree_t_enc, workex_enc,
        etest_p, spec_enc, mba_p
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # ---- Show Result ----
    st.markdown('## 📊 Your Result')
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.success('🎉 HIGH CHANCE OF PLACEMENT!')
            st.metric('Placement Probability', f'{probability[1]*100:.1f}%')
        else:
            st.error('⚠️ LOW CHANCE OF PLACEMENT')
            st.metric('Placement Probability', f'{probability[1]*100:.1f}%')

    with res_col2:
        # Simple bar chart
        fig, ax = plt.subplots(figsize=(4, 2))
        bars = ax.bar(['Not Placed', 'Placed'],
                     [probability[0]*100, probability[1]*100],
                     color=['#C00000', '#70AD47'])
        ax.set_ylabel('Probability %')
        ax.set_ylim(0, 100)
        ax.set_title('Placement Probability')
        for bar, val in zip(bars, [probability[0]*100, probability[1]*100]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                   f'{val:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)

    # ---- Improvement Tips ----
    st.markdown('---')
    st.markdown('## 💡 What To Improve')

    tips = []
    if degree_p < 65:
        tips.append('📚 Your degree percentage is below 65%. Focus on academics.')
    if workex == 'No':
        tips.append('💼 Get an internship! Work experience significantly boosts placement.')
    if etest_p < 60:
        tips.append('💻 Practice aptitude tests and coding platforms like HackerRank.')
    if mba_p < 60:
        tips.append('📊 Improve your MBA scores - work on analytical and management skills.')

    if not tips:
        tips.append('✅ Your profile looks strong! Focus on interview preparation.')

    for tip in tips:
        st.info(tip)

    st.markdown('---')
    st.caption('Note: This is a predictive model. Results are based on historical patterns.')

# ---- Sidebar Info ----
with st.sidebar:
    st.header('About This App')
    st.write('This ML model predicts campus placement chances based on academic and personal data.')
    st.write('Built with: Python, XGBoost, Streamlit')
    st.markdown('---')
    st.write('**Tech Stack:**')
    st.write('• XGBoost Classifier')
    st.write('• Scikit-learn')
    st.write('• SHAP Explainability')
    st.write('• Streamlit')
