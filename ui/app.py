import streamlit as st
import sys
import os
import html

# Add parent directory to path to import model_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service import get_model_service

st.set_page_config(page_title="SentiMind: Mental Health AI", page_icon="🧠", layout="wide")

def get_theme_css(dark_mode: bool) -> str:
    app_bg = "#0f172a" if dark_mode else "#f0f9ff"
    card_bg = "#111827" if dark_mode else "#ffffff"
    text_muted = "#cbd5e1" if dark_mode else "#64748b"
    history_border = "#1f2937" if dark_mode else "#eef2f7"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {app_bg};
        }}

        .logo-text {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #0369a1;
        }}

        .sentiment-badge {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .sentiment-positive {{ color: #059669; }}
        .sentiment-negative {{ color: #dc2626; }}
        .sentiment-neutral {{ color: #4b5563; }}

        .mh-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-block;
        }}
        .mh-stable {{ background-color: #d1fae5; color: #065f46; }}
        .mh-alert {{ background-color: #fee2e2; color: #991b1b; }}

        .confidence-text {{
            font-size: 0.75rem;
            color: {text_muted};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .confidence-value {{
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .history-item {{
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid {history_border};
            margin-bottom: 0.75rem;
            background-color: {card_bg};
        }}

        .score-circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 10px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            margin: 0 auto;
        }}

        .score-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        /* Custom button styling */
        div.stButton > button[data-testid="stBaseButton-primary"] {{
            background-color: #0369a1 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 2rem !important;
        }}

        /* Hide streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
    """


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "history" not in st.session_state:
    st.session_state.history = []
if "total_analyses" not in st.session_state:
    st.session_state.total_analyses = 0

st.markdown(get_theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Top Header
head_col1, head_col2, head_col3 = st.columns([4, 1, 1])
with head_col1:
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 2rem;">🧠</span><span class="logo-text">SentiMind: Mental Health & Sentiment AI</span></div>', unsafe_allow_html=True)
with head_col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_analyses = 0
        st.rerun()

# Main Layout
main_col, history_col = st.columns([2.5, 1])

service = get_model_service()

with main_col:
    # Analyze Card
    with st.container(border=True):
        st.markdown("""
        <h3 style="margin-top: 0; font-size: 1.25rem;">Mental Health Check-in</h3>
        <p style="color: #64748b; font-size: 0.9rem;">Describe how you're feeling today. We'll analyze both the sentiment and potential mental health indicators.</p>
        """, unsafe_allow_html=True)

        user_input = st.text_area("Input", placeholder="e.g., I've been feeling very overwhelmed lately and can't stop worrying...", label_visibility="collapsed", height=100)
        analyze_clicked = st.button("Analyze Wellbeing", type="primary")

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please share your thoughts.")
        elif service.model is None:
            st.error("Model is not available. Please ensure it is trained.")
        else:
            with st.spinner("Processing emotional patterns..."):
                result = service.predict(user_input)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.total_analyses += 1
                    safe_input = html.escape(user_input)
                    st.session_state.history.insert(0, {
                        "text": safe_input[:100] + "..." if len(safe_input) > 100 else safe_input,
                        "sentiment": result["overall_sentiment"],
                        "mh": result["mental_health_indicator"]
                    })

                    st.markdown("### Analysis Report")

                    # Report Grid
                    col_sent, col_mh, col_score = st.columns([1.2, 1.2, 1])

                    with col_sent:
                        with st.container(border=True):
                            sentiment_class = f"sentiment-{result['overall_sentiment'].lower()}"
                            icon = "😊" if result['overall_sentiment'] == "Positive" else "😔" if result['overall_sentiment'] == "Negative" else "😐"
                            st.markdown(f"""
                                <div class="confidence-text">Overall Sentiment</div>
                                <div class="sentiment-badge {sentiment_class}">{icon} {result['overall_sentiment']}</div>
                                <div style="margin-top: 10px;" class="confidence-text">Confidence: {result['top_prediction']['confidence']:.0%}</div>
                            """, unsafe_allow_html=True)

                    with col_mh:
                        with st.container(border=True):
                            mh_class = "mh-alert" if result['mh_alert'] else "mh-stable"
                            st.markdown(f"""
                                <div class="confidence-text">Mental Health Indicator</div>
                                <div style="margin-top: 10px;">
                                    <span class="mh-badge {mh_class}">{result['mental_health_indicator']}</span>
                                </div>
                                <div style="margin-top: 18px; font-size: 0.8rem; color: #64748b;">Detected from patterns</div>
                            """, unsafe_allow_html=True)

                    with col_score:
                        with st.container(border=True):
                            score = result['sentiment_score']
                            score_color = "#059669" if score > 0.1 else "#dc2626" if score < -0.1 else "#4b5563"
                            st.markdown(f"""
                                <div class="confidence-text" style="text-align:center;">Score</div>
                                <div class="score-circle" style="border-color: {score_color}44; border-top-color: {score_color};">
                                    <div class="score-value" style="color: {score_color};">{score:.2f}</div>
                                </div>
                            """, unsafe_allow_html=True)

                    # Detailed Insights
                    with st.container(border=True):
                        st.markdown(f"**Insight:** {html.escape(result['key_logic'])}")

                        # Show all significant labels
                        st.markdown("---")
                        st.markdown("**Detected Emotional Traits:**")
                        cols = st.columns(3)
                        significant_labels = [r for r in result['predictions'] if r['confidence'] > 0.3][:6]
                        for i, r in enumerate(significant_labels):
                            cols[i % 3].markdown(f"**{r['label'].capitalize()}**: `{r['confidence']:.1%}`")

with history_col:
    st.markdown("#### Recent Checks")
    for item in st.session_state.history[:10]:
        with st.container():
            st.markdown(f"""
            <div class="history-item">
                <div style="font-size: 0.85rem; font-weight: 600;">{item['sentiment']} | {item['mh']}</div>
                <div style="font-size: 0.8rem; color: #64748b; font-style: italic; margin-top: 5px;">"{item['text']}"</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 20px;">
            Total Analyses: {st.session_state.total_analyses}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")
