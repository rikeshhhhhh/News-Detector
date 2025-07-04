import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model and label map (cache for efficiency)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("news_classifier_model.pkl")
        label_map = {0: "FAKE", 1: "REAL"}  # Adjust if your model uses other labels
        return model, label_map
    except FileNotFoundError:
        st.error("Model file 'news_classifier_model.pkl' not found.")
        return None, None

model, label_map = load_model()

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Custom CSS
st.markdown("""
    <style>
        .boxed-section {
            border: 1px solid #444;
            background-color: #1a1a1a;
            padding: 24px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 0 10px #111;
        }
        .history-entry {
            background-color: #222;
            color: #ddd;
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 5px;
            font-size: 13px;
        }
        .history-entry b {
            color: #fff;
        }
        .toggle-btn {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #2a2a2a;
            color: #eee;
            padding: 15px;
            margin-top: 15px;
            border-left: 5px solid #00cc99;
            border-radius: 5px;
            font-size: 16px;
        }
        .export-btn {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown("<h1 style='text-align: center; color: #00cc99;'>NEWS CLASSIFIER</h1>", unsafe_allow_html=True)

# Classifier Box
st.markdown("<div class='boxed-section'>", unsafe_allow_html=True)
st.markdown("### Classify a News Article")

user_input = st.text_area("ENTER TEXT BELOW", height=180, placeholder="e.g. India’s GDP grows by 7.8% in Q1 2025")

if model and label_map:
    if st.button("CLASSIFY"):
        if user_input.strip():
            with st.spinner("Classifying..."):
                try:
                    prediction = model.predict([user_input])[0]
                    predicted_label = label_map[int(prediction)]

                    try:
                        proba = model.predict_proba([user_input])[0]
                        confidence = max(proba)
                    except:
                        confidence = None

                    # Store in history with timestamp
                    st.session_state.history.append({
                        "text": user_input,
                        "label": predicted_label,
                        "confidence": confidence or 0.0,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Display result
                    result = f"{predicted_label} NEWS"
                    confidence_text = f"<div style='margin-top: 10px;'>Confidence: <b>{confidence:.2f}</b></div>" if confidence else ""

                    st.markdown(f"""
                        <div class='result-box'>
                            Prediction: {result.upper()}
                            {confidence_text}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Classification failed: {str(e)}")
        else:
            st.warning("Please enter some text to classify.")
st.markdown("</div>", unsafe_allow_html=True)

# Toggle History Button
st.markdown("<div class='toggle-btn'>", unsafe_allow_html=True)
if st.button("Show History" if not st.session_state.show_history else "Hide History"):
    st.session_state.show_history = not st.session_state.show_history
st.markdown("</div>", unsafe_allow_html=True)

# History Box (if toggled on)
if st.session_state.show_history and model and label_map:
    st.markdown("<div class='boxed-section'>", unsafe_allow_html=True)
    st.markdown("### History (Last 10)")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.markdown(f"""
                <div class="history-entry">
                    <b>{item['label']} ({item['confidence']*100:.1f}%)</b> - {item['timestamp']}<br>
                    {item['text'][:100]}...
                </div>
            """, unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared.")
        
        # Export history
        if st.button("Export History"):
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="classification_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No history yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# Optional feedback for model retraining
if model and label_map and st.session_state.history:
    st.markdown("<div class='boxed-section'>", unsafe_allow_html=True)
    st.markdown("### Provide Feedback")
    feedback = st.selectbox("Was the last prediction correct?", ["", "Yes", "No"])
    if feedback == "No" and st.button("Submit Feedback"):
        last_item = st.session_state.history[-1]
        st.session_state.history[-1]["feedback"] = "Incorrect"
        st.success("Feedback submitted. This can be used for model retraining.")
    st.markdown("</div>", unsafe_allow_html=True)