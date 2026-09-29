import pandas as pd
import plotly.express as px
import streamlit as st
from src.audio_processor import extract_mel_spectrogram

st.set_page_config(page_title="Researcher Diagnostic Studio", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .card-panel {
            background: #FFFFFF;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 18px;
            padding: 1.25rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-low { background: rgba(16, 185, 129, 0.12); color: #047857; }
        .status-moderate { background: rgba(245, 158, 11, 0.14); color: #b45309; }
        .status-high { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Researcher Diagnostic Studio")
st.caption("Audio review, Mel-spectrogram analysis, attention diagnostics, and logged sessions")

uploaded_audio = st.file_uploader(
    "Upload recorded audio",
    type=["wav", "mp3", "m4a", "ogg"],
    help="Use a speech sample to inspect model features and fatigue prediction output.",
)

mel_spectrogram = None
if uploaded_audio is not None:
    st.audio(uploaded_audio)
    try:
        mel_spectrogram = extract_mel_spectrogram(uploaded_audio)
    except Exception as exc:
        st.error(f"Audio processing failed: {exc}")

left, right = st.columns([1.5, 1])

with left:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.subheader("Audio features")
    if uploaded_audio is None:
        st.info("Upload an audio recording to generate its Mel-spectrogram.")
    elif mel_spectrogram is not None:
        st.success("Audio processed: 16 kHz mono, voice activity trimmed, and padded or clipped to 15 seconds.")
        st.caption(f"Mel-spectrogram shape: {mel_spectrogram.shape[0]} bins × {mel_spectrogram.shape[1]} frames")
    st.markdown('</div>', unsafe_allow_html=True)

    if mel_spectrogram is not None:
        fig = px.imshow(
            mel_spectrogram,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels={"x": "Time Frames", "y": "Mel Bins", "color": "Power (dB)"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.subheader("Fatigue classification")
    st.info("Prediction and attention analysis are unavailable because a trained model is not connected yet.")
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Example session data")
st.caption("Illustrative rows only; these are not participant records.")

session_rows = [
    {
        "session_id": "A-001",
        "respondent_id": "WVSU_CS_001",
        "task_level": "Easy",
        "ground_truth_score": 3,
        "predicted_fatigue": "Low",
    },
    {
        "session_id": "A-002",
        "respondent_id": "WVSU_CS_002",
        "task_level": "Moderate",
        "ground_truth_score": 5,
        "predicted_fatigue": "Moderate",
    },
    {
        "session_id": "A-003",
        "respondent_id": "WVSU_CS_003",
        "task_level": "Intensive",
        "ground_truth_score": 6,
        "predicted_fatigue": "High",
    },
]

session_df = pd.DataFrame(session_rows)
filtered_df = st.dataframe(session_df, use_container_width=True)

csv = session_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download dataset (.CSV)",
    data=csv,
    file_name="fatigue_session_logs.csv",
    mime="text/csv",
)
