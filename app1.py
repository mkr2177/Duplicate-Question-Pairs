import streamlit as st
import helper
import pickle
import numpy as np
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# --- Style ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(120deg, #f0f8ff, #e6f2ff);
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput input, .stTextArea textarea {
        font-size: 16px;
        padding: 10px;
    }
    .result-badge {
        font-size: 28px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        padding: 10px 20px;
        border-radius: 10px;
        margin-top: 20px;
        display: inline-block;
    }
    .not-duplicate {
        background-color: #FF4136;
    }
    .confidence {
        font-size: 18px;
        margin-top: 10px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align:center;'>🧠 Duplicate Question Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Check if two questions mean the same — one at a time or upload a CSV file!</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📤 Bulk Prediction via CSV"])

# ------------------- TAB 1: SINGLE -------------------
with tab1:
    q1 = st.text_area("✍️ Question 1", height=100)
    q2 = st.text_area("✍️ Question 2", height=100)

    col1, col2 = st.columns([1, 1])
    with col1:
        predict = st.button("🚀 Predict")
    with col2:
        reset = st.button("🔄 Reset")

    if reset:
        st.experimental_rerun()

    if predict:
        if q1.strip() == "" or q2.strip() == "":
            st.warning("⚠️ Please enter both questions.")
        else:
            # Auto-correct spelling
            q1 = str(TextBlob(q1).correct())
            q2 = str(TextBlob(q2).correct())

            query = helper.query_point_creator(q1, q2)
            proba = model.predict_proba(query)[0]
            result = np.argmax(proba)
            confidence = np.max(proba) * 100

            if result == 1:
                st.markdown("<div class='result-badge'>✅ Duplicate</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-badge not-duplicate'>❌ Not Duplicate</div>", unsafe_allow_html=True)

            st.markdown(f"<p class='confidence'>📊 Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

# ------------------- TAB 2: BULK -------------------
with tab2:
    uploaded_file = st.file_uploader("📄 Upload CSV with columns: `question1`, `question2`", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'question1' not in df.columns or 'question2' not in df.columns:
            st.error("CSV must contain 'question1' and 'question2' columns.")
        else:
            predictions = []
            probs = []

            st.info(f"Processing {len(df)} pairs...")

            for idx, row in df.iterrows():
                q1 = str(TextBlob(row['question1']).correct())
                q2 = str(TextBlob(row['question2']).correct())
                query = helper.query_point_creator(q1, q2)
                proba = model.predict_proba(query)[0]
                pred = np.argmax(proba)
                predictions.append("Duplicate" if pred == 1 else "Not Duplicate")
                probs.append(round(np.max(proba) * 100, 2))

            df['Prediction'] = predictions
            df['Confidence (%)'] = probs

            st.dataframe(df[['question1', 'question2', 'Prediction', 'Confidence (%)']])

            # Plot
            st.markdown("### 📊 Confidence Chart")
            fig, ax = plt.subplots(figsize=(10, len(df) * 0.4))
            colors = ['green' if p == "Duplicate" else 'red' for p in df['Prediction']]
            ax.barh(df.index, df['Confidence (%)'], color=colors)
            ax.set_yticks(df.index)
            ax.set_yticklabels(df['Prediction'])
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Prediction Confidence for Each Pair")
            st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
