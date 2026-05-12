import os
import numpy as np
import librosa
import gradio as gr
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, GlobalAveragePooling2D,
    Bidirectional, LSTM, Dense, Dropout, Activation
)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2

# =========================
# CONFIG
# =========================
METRICS = ["Fluency", "Clarity", "Pacing", "Engagement", "Confidence"]
WEIGHTS_PATH = "SS_AI.weights.h5"

# =========================
# BASE MODEL (CNN)
# =========================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# =========================
# MODEL
# =========================
def build_model():
    model = Sequential()

    model.add(TimeDistributed(base_model, input_shape=(None, 224, 224, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.3))

    # 12 dense layers
    model.add(Dense(256, kernel_regularizer=l2(0.001)))
    model.add(Activation("relu"))

    for _ in range(4):
        model.add(Dense(256))
        model.add(Activation("relu"))

    for _ in range(7):
        model.add(Dense(128))
        model.add(Activation("relu"))

    model.add(Dense(5, activation="sigmoid"))

    return model


model = build_model()

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("✅ Weights loaded successfully")
else:
    print("⚠️ No weights file found")


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(audio):

    if audio is None:
        raise ValueError("No audio provided")

    # =========================
    # HANDLE GRADIO INPUT TYPES
    # =========================

    # Case 1: (sr, numpy array)
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, y = audio

    # Case 2: file path (Gradio type="filepath")
    elif isinstance(audio, str):
        y, sr = librosa.load(audio, sr=None)

    else:
        raise ValueError(f"Unsupported audio format: {type(audio)}")

    # =========================
    # FORCE CLEAN NUMPY FLOAT AUDIO
    # =========================
    y = np.array(y, dtype=np.float32)

    # mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # resample safely
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64)
    spec = librosa.power_to_db(spec)

    spec = np.resize(spec, (224, 224))
    spec = np.stack([spec, spec, spec], axis=-1)

    # IMPORTANT: match model input shape
    return np.expand_dims(np.expand_dims(spec, axis=0), axis=0)


# =========================
# FEEDBACK
# =========================
def feedback(score):
    if score >= 0.7:
        return "🟢 Excellent"
    elif score >= 0.4:
        return "🟡 Good / Improve"
    else:
        return "🔴 Needs Work"


# =========================
# PREDICTION
# =========================
def predict(audio):

    if audio is None:
        return "No audio input"

    features = extract_features(audio)
    preds = model.predict(features, verbose=0)[0]

    output = "## 🎤 Speech Analysis Results\n"

    for i, m in enumerate(METRICS):
        score = float(preds[i])

        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))

        output += f"""
### {m}
**Score:** {score:.3f}

{bar}

**Feedback:** {feedback(score)}

---
"""

    return output


# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="SpeakSmart AI") as SS_AI_LD:

    gr.Markdown("# 🎤 SpeakSmart AI Dashboard")
    gr.Markdown("Upload or record speech and get AI feedback")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="numpy",
            label="Input Audio"
        )

        playback = gr.Audio(label="Playback")

    analyze_btn = gr.Button("Analyze Speech")

    # ✅ FIXED OUTPUT (THIS WAS YOUR BUG)
    output = gr.Markdown(label="Performance Results")

    analyze_btn.click(
        fn=predict,
        inputs=audio_input,
        outputs=output
    )

    audio_input.change(
        fn=lambda x: x,
        inputs=audio_input,
        outputs=playback
    )


SS_AI_LD.launch(inbrowser=True)