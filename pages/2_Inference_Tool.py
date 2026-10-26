import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.audio_processor import TARGET_DURATION, extract_mel_spectrogram, prepare_model_tensor
from src.database import fetch_all_sessions

st.set_page_config(page_title="Researcher Diagnostic Studio", layout="wide")

CLASS_DETAILS = [
    {
        "label": "Low Fatigue",
        "hiligaynon": "Gamay lang nga kapoy",
        "scale": "Samn-Perelli 1–2",
        "description": "Rested, alert state.",
    },
    {
        "label": "Moderate Fatigue",
        "hiligaynon": "Sakto-sakto nga kapoy",
        "scale": "Samn-Perelli 3–4",
        "description": "Mild fatigue, reduced freshness.",
    },
    {
        "label": "High Fatigue",
        "hiligaynon": "Grabe nga kapoy",
        "scale": "Samn-Perelli 5–7",
        "description": "Pronounced mental exhaustion, difficulty concentrating.",
    },
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("FATIGUE_MODEL_PATH", "fatigue_cnn_attention.keras"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH


@st.cache_resource(show_spinner="Loading fatigue detection model...")
def load_inference_model(model_path: str):
    import tensorflow as tf

    from src.model_architecture import TemporalAttention, get_inference_model

    trained_model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )
    return get_inference_model(trained_model)


st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { color: #1D4ED8; }
        h1 { border-left: 5px solid #2563EB; padding-left: 0.75rem; }
        .card-panel {
            background: #FFFFFF;
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-top: 4px solid #2563EB;
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
        div[data-testid="stFileUploaderDropzone"] {
            border: 1px dashed #60A5FA;
            border-radius: 14px;
            background: #EFF6FF;
        }
        div[data-testid="stFileUploaderDropzone"]:hover { border-color: #2563EB; }
        div.stButton > button[kind="primary"] {
            background: #2563EB;
            border-color: #2563EB;
            color: #FFFFFF;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1D4ED8;
            border-color: #1D4ED8;
        }
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
prediction = None
attention = None
peak_attention_time = None
if uploaded_audio is not None:
    st.audio(uploaded_audio)
    try:
        with st.spinner("Processing audio and extracting Mel-spectrogram..."):
            mel_spectrogram = extract_mel_spectrogram(uploaded_audio)
    except Exception as exc:
        st.error(f"Audio processing failed: {exc}")
    else:
        if MODEL_PATH.is_file():
            try:
                inference_model = load_inference_model(str(MODEL_PATH))
                model_input = prepare_model_tensor(mel_spectrogram)
                class_probabilities, attention_weights = inference_model.predict(
                    model_input,
                    verbose=0,
                )
                class_probabilities = np.asarray(class_probabilities)
                attention_weights = np.asarray(attention_weights)

                if class_probabilities.shape != (1, 3):
                    raise ValueError(
                        "The model must return probabilities for Low, Moderate, and High fatigue."
                    )
                if (
                    np.any(class_probabilities < 0)
                    or np.any(class_probabilities > 1)
                    or not np.isclose(class_probabilities.sum(), 1.0, atol=1e-3)
                ):
                    raise ValueError("The model returned invalid class confidence scores.")
                if attention_weights.ndim != 2 or attention_weights.shape[0] != 1:
                    raise ValueError("The model returned an invalid temporal attention shape.")
                if (
                    attention_weights.shape[1] == 0
                    or np.any(attention_weights < 0)
                    or not np.isclose(attention_weights.sum(), 1.0, atol=1e-3)
                ):
                    raise ValueError("The model returned invalid temporal attention weights.")
                if (
                    not np.isfinite(class_probabilities).all()
                    or not np.isfinite(attention_weights).all()
                ):
                    raise ValueError("The model returned non-finite inference values.")

                prediction = class_probabilities[0]
                attention = attention_weights[0]
                peak_attention_index = int(np.argmax(attention))
                peak_attention_time = (
                    (peak_attention_index + 0.5) * TARGET_DURATION / len(attention)
                )
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                st.error(f"Model analysis failed: {exc}")

left, right = st.columns([1.5, 1])

with left:
    with st.container(border=True):
        st.subheader("Audio features")
        if uploaded_audio is None:
            st.info("Upload an audio recording to generate its Mel-spectrogram.")
        elif mel_spectrogram is not None:
            st.success("Audio processed: 16 kHz mono, voice activity trimmed, and padded or clipped to 15 seconds.")
            st.caption(f"Mel-spectrogram shape: {mel_spectrogram.shape[0]} bins × {mel_spectrogram.shape[1]} frames")

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
    with st.container(border=True):
        st.subheader("Fatigue classification")
        if uploaded_audio is None:
            st.info("Upload an audio recording to run fatigue classification.")
        elif not MODEL_PATH.is_file():
            st.warning(
                "A trained model is not available. Add "
                "`fatigue_cnn_attention.keras` to the project root or set "
                "`FATIGUE_MODEL_PATH` to the model file to enable classification, "
                "confidence scores, and temporal attention."
            )
        elif prediction is not None:
            predicted_index = int(np.argmax(prediction))
            predicted_class = CLASS_DETAILS[predicted_index]
            st.metric("Predicted fatigue level", predicted_class["label"])
            st.metric("Prediction confidence", f"{prediction[predicted_index]:.1%}")
            st.caption("Interactive class probabilities (hover over a bar for the exact score)")
            confidence_df = pd.DataFrame(
                {
                    "Fatigue level": [item["label"] for item in CLASS_DETAILS],
                    "Probability": prediction,
                }
            )
            confidence_fig = px.bar(
                confidence_df,
                x="Probability",
                y="Fatigue level",
                orientation="h",
                color="Fatigue level",
                color_discrete_sequence=["#10B981", "#F59E0B", "#EF4444"],
                category_orders={
                    "Fatigue level": [
                        item["label"] for item in reversed(CLASS_DETAILS)
                    ]
                },
                text=confidence_df["Probability"].map(lambda value: f"{value:.1%}"),
                hover_data={"Probability": ":.2%", "Fatigue level": True},
            )
            confidence_fig.update_layout(
                xaxis=dict(title="Model probability", range=[0, 1], tickformat=".0%"),
                yaxis_title=None,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            confidence_fig.update_traces(
                textposition="outside",
                hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            )
            st.plotly_chart(confidence_fig, use_container_width=True)

if prediction is not None:
    with st.container(border=True):
        st.subheader("Fatigue scale reference")
        detail_columns = st.columns(3)
        for column, detail in zip(detail_columns, CLASS_DETAILS):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{detail['label']}**")
                    st.caption(detail["hiligaynon"])
                    st.markdown(f"`{detail['scale']}`")
                    st.caption(detail["description"])

if attention is not None:
    with st.container(border=True):
        st.subheader("Temporal attention")
        st.metric(
            "Peak attention timestamp",
            f"{peak_attention_time:.3f} s",
            help="Center timestamp of the model attention bin with the highest relative weight.",
        )
        attention_times = (
            np.arange(len(attention), dtype=np.float32) + 0.5
        ) * TARGET_DURATION / len(attention)
        attention_df = pd.DataFrame(
            {"Time (seconds)": attention_times, "Attention weight": attention}
        )
        attention_fig = px.area(
            attention_df,
            x="Time (seconds)",
            y="Attention weight",
            labels={"Attention weight": "Relative importance"},
            title="Model attention across the 15-second audio window",
        )
        attention_fig.add_scatter(
            x=[peak_attention_time],
            y=[attention[int(np.argmax(attention))]],
            mode="markers",
            name="Peak attention",
            marker={"color": "#DC2626", "size": 10, "symbol": "diamond"},
            hovertemplate=(
                f"Peak attention: {peak_attention_time:.3f} s"
                "<br>Relative weight: %{y:.4f}<extra></extra>"
            ),
        )
        attention_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(attention_fig, use_container_width=True)
        st.caption(
            "The timestamp marks the center of the highest-weight attention bin. "
            "Attention weights show relative model focus, not calibrated confidence "
            "or clinical proof that fatigue markers occurred at that instant."
        )

with st.expander("Researcher interpretation guide"):
    st.markdown(
        """
        **Mel-spectrogram (128 Mel bins):** The vertical axis groups acoustic
        energy into 128 perceptually spaced frequency bands; the horizontal axis
        represents time across the standardized 15-second clip. Brighter or
        higher-energy regions indicate stronger signal energy in those bands.
        This view is a model input representation, not a transcript or a direct
        measurement of fatigue.

        **Temporal attention:** The curve shows how the model distributes its
        relative attention across time steps while forming its prediction. The
        peak timestamp is the center of the bin with the largest attention
        weight. Attention is model-relative and should not be interpreted as a
        calibrated confidence score, causal explanation, or clinically validated
        localization of fatigue.

        **Class probabilities:** The three bars show the model's output
        probabilities for the Low, Moderate, and High classes. They are model
        scores, not a diagnosis. The Samn–Perelli ranges shown in the reference
        cards explain the study's class mapping.
        """
    )

st.subheader("Live session table")

session_df = pd.DataFrame(columns=["session_id", "respondent_id", "task_level", "ground_truth_score", "predicted_fatigue", "audio_storage_url"])
try:
    with st.spinner("Loading live session records..."):
        fetched_sessions = fetch_all_sessions()
    if fetched_sessions.empty:
        st.info("No participant sessions have been logged yet. Once a participant completes a task, their session row will appear here.")
    else:
        session_df = fetched_sessions.copy()
except Exception as exc:
    st.info("Live session data is unavailable until the Supabase credentials are configured. Once the database is connected, this table will populate automatically.")
    st.caption(f"Connection status: {exc}")

st.dataframe(session_df, use_container_width=True)

csv = session_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download dataset (.CSV)",
    data=csv,
    file_name="fatigue_session_logs.csv",
    mime="text/csv",
    disabled=session_df.empty,
)
