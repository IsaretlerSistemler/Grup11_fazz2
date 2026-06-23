import streamlit as st
import librosa
import pickle
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from feature_extractor import extract_features

st.set_page_config(page_title="Emotion AI", layout="centered")

# ================= UI STYLE =================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        -45deg,
        #ff0844,
        #ff4d6d,
        #c9184a,
        #7209b7,
        #560bad
    );
    background-size: 500% 500%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1 {
    color: #ffffff !important;
    text-align: center;
    font-size: 3rem !important;
    font-weight: 900 !important;
    text-shadow: 0px 4px 20px rgba(255,255,255,0.25);
}

h2, h3 {
    color: #ffe5ec !important;
    text-align: center;
    font-weight: 700 !important;
}

p, label, div {
    color: #fff5f7 !important;
}

/* RESULT CARD */
.result-box {

    padding: 40px;
    border-radius: 35px;

    background: rgba(255,255,255,0.10);

    backdrop-filter: blur(25px);

    border: 1px solid rgba(255,255,255,0.20);

    text-align: center;

    margin-top: 20px;

    animation: floatCard 5s ease-in-out infinite;

    box-shadow:
        0 0 25px rgba(255,0,120,0.5),
        0 0 60px rgba(114,9,183,0.35),
        0 20px 50px rgba(0,0,0,0.4);
}

@keyframes floatCard {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-8px);}
    100% {transform: translateY(0px);}
}

.emoji {
    font-size: 100px;
}

.label {
    font-size: 48px;
    font-weight: 900;
    color: white;
    text-transform: uppercase;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 15px;
}

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#ff0844,#7209b7);
}

</style>
""", unsafe_allow_html=True)

# ================= MODEL =================
@st.cache_resource
def load_model():
    with open("emotion_model_faz2.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()

model = pipeline["model"]
scaler = pipeline["scaler"]
selector = pipeline["selector"]
le = pipeline["label_encoder"]

# ================= SIDEBAR INFO =================
st.sidebar.title("📊 Model Info")

st.sidebar.success("Accuracy: ~81% (base model)")
st.sidebar.write(f"Feature size: {scaler.n_features_in_}")

# ================= EMOJI MAP =================
emoji_map = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲"
}

# ================= UI =================
st.title("🎙️ Emotion Detection AI")
st.write("Ses yükle → duygu analizini gör")

file = st.file_uploader("WAV dosyası", type=["wav"])

# ================= FILE INFO =================
if file:
    st.sidebar.markdown("### 📁 Upload Info")
    st.sidebar.write(f"Dosya: {file.name}")
    st.sidebar.write(f"Boyut: {round(file.size/1024,2)} KB")

    st.audio(file, format="audio/wav")

    with st.spinner("Analiz ediliyor..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            path = tmp.name

        try:
            y, sr = librosa.load(path, sr=22050, mono=True)

            feat = extract_features(y, sr)
            feat = feat.reshape(1, -1)

            feat = scaler.transform(feat)
            feat = selector.transform(feat)

            pred = model.predict(feat)[0]

            label = le.inverse_transform([pred])[0]

            probas = model.predict_proba(feat)[0]

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
            st.stop()

        finally:
            if os.path.exists(path):
                os.remove(path)

    emoji = emoji_map.get(label.lower(), "🎧")

    # ================= RESULT =================
    st.markdown(f"""
    <div class="result-box">
        <div class="emoji">{emoji}</div>
        <div class="label">{label}</div>
        <p>AI tarafından tespit edilen duygu</p>
    </div>
    """, unsafe_allow_html=True)

    # ================= CONFIDENCE =================
    st.subheader("📊 Duygu Dağılımı")

    classes = le.classes_

    sorted_probs = sorted(
        zip(classes, probas),
        key=lambda x: x[1],
        reverse=True
    )

    for cls, prob in sorted_probs:
        st.write(f"{cls} ({prob*100:.1f}%)")
        st.progress(float(prob))

    # ================= TOP 3 =================
    st.subheader("🔥 En Güçlü 3 Tahmin")

    for cls, prob in sorted_probs[:3]:
        st.write(
            f"{emoji_map.get(cls.lower(),'🎧')} {cls} → %{prob*100:.1f}"
        )

    st.success("Analiz tamamlandı ✨")

# ================= CONFUSION MATRIX (SAFE) =================
# ================= CONFUSION MATRIX (UI MATCHED) =================

st.subheader("🧮 Confusion Matrix")

try:
    X_test = pipeline.get("X_test", None)
    y_test = pipeline.get("y_test", None)

    if X_test is not None and y_test is not None:

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="magma",   # 🔥 UI ile uyumlu neon renk
            linewidths=1,
            linecolor="white",
            cbar=True,
            square=True,
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=ax
        )

        ax.set_title(
            "Emotion AI - Confusion Matrix",
            fontsize=14,
            color="white",
            pad=20
        )

        ax.set_xlabel("Predicted Label", color="white")
        ax.set_ylabel("True Label", color="white")

        ax.tick_params(colors="white")

        fig.patch.set_facecolor('#1a001f')  # koyu mor arka plan
        ax.set_facecolor('#1a001f')

        st.pyplot(fig)

    else:
        st.info("Confusion matrix için test dataset bulunamadı.")

except Exception as e:
    st.warning(f"Confusion matrix yüklenemedi: {e}") 