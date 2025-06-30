import streamlit as st
import helper
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# --- Custom Background & Style ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f6f9fc, #e9f2ff);
        padding: 20px;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 10px;
    }
    .result-badge {
        font-size: 28px;
        font-weight: bold;
        color: white;
        background-color: #4CAF50;
        padding: 10px 20px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 20px;
    }
    .not-duplicate {
        background-color: #FF4136;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #003366;'>🧠 Duplicate Question Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter two questions to check if they are semantically the same</p>", unsafe_allow_html=True)

# --- Input ---
col1, col2 = st.columns(2)
with col1:
    q1 = st.text_area("✍️ Question 1", height=100)
with col2:
    q2 = st.text_area("✍️ Question 2", height=100)

# --- Predict Button ---
if st.button("🚀 Predict"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("⚠️ Please enter both questions to proceed.")
    else:
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]

        if result:
            st.markdown("<div class='result-badge'>✅ Duplicate</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-badge not-duplicate'>❌ Not Duplicate</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<small style='color: gray;'>Made with ❤️ using Streamlit</small>", unsafe_allow_html=True)
