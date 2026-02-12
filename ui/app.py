import streamlit as st
import sys
import os

# Add parent directory to path to import model_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service import get_model_service

st.set_page_config(page_title="Custom Sentiment Analysis", page_icon="😊")

st.title("🎭 Custom Sentiment Analysis")
st.markdown("Fine-tuned DistilBERT for emotion detection.")

# Sidebar for model status
st.sidebar.title("Model Info")
service = get_model_service()

if service.model is None:
    st.sidebar.error("Model not trained or loaded.")
    if st.sidebar.button("Try Reloading"):
        service = get_model_service(force_reload=True)
        st.rerun()
else:
    st.sidebar.success("Model ready!")
    st.sidebar.info(f"Model Path: `{service.model_path}`")
    if service.label_mapping:
        st.sidebar.write("Labels:", list(service.label_mapping['label2id'].keys()))

# Main UI
user_input = st.text_area("Enter text to analyze:", placeholder="I feel so hopeful today...")

if st.button("Predict Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    elif service.model is None:
        st.error("Model is not available. Please run training first.")
    else:
        with st.spinner("Analyzing..."):
            result = service.predict(user_input)

            if "error" in result:
                st.error(result["error"])
            else:
                top_pred = result["top_prediction"]

                # Big result display
                st.subheader(f"Top Sentiment: {top_pred['label'].capitalize()}")
                st.write(f"Confidence: **{top_pred['confidence']:.2%}**")
                st.progress(top_pred['confidence'])

                # Detailed breakdown
                st.markdown("---")
                st.subheader("Emotion Breakdown")

                for pred in result["predictions"]:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(pred["label"])
                    with col2:
                        st.progress(pred["confidence"])
                        st.caption(f"Score: {pred['confidence']:.4f}")

st.markdown("---")
st.caption("Built with HuggingFace Transformers, FastAPI, and Streamlit.")
