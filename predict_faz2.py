import sys
import pickle
import librosa
import warnings

from feature_extractor import extract_features

warnings.filterwarnings("ignore")


# ============================================================
# PREDICT
# ============================================================

def predict(audio_path):

    with open("emotion_model_faz2.pkl", "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    scaler = data["scaler"]
    selector = data["selector"]
    le = data["label_encoder"]

    try:
        y, sr = librosa.load(
            audio_path,
            sr=22050,
            mono=True
        )

    except Exception as e:

        print(f"❌ Ses dosyası okunamadı: {e}")
        return

    features = extract_features(y, sr)

    if features is None:

        print("❌ Feature çıkarılamadı")
        return

    # Scale
    features_sc = scaler.transform(
        features.reshape(1, -1)
    )

    # Feature Selection
    features_sc = selector.transform(
        features_sc
    )

    # Prediction
    pred_idx = model.predict(features_sc)[0]

    pred_label = le.inverse_transform(
        [pred_idx]
    )[0]

    # Probabilities
    probas = model.predict_proba(
        features_sc
    )[0]

    print(f"\n🎤 DOSYA : {audio_path}")

    print(f"\n🎯 TAHMİN : {pred_label.upper()}")

    print("\n📊 OLASILIKLAR:\n")

    for cls, prob in zip(le.classes_, probas):

        bar = "█" * int(prob * 30)

        print(
            f"{cls:12s} "
            f"{bar:<30} "
            f"{prob*100:5.1f}%"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nKullanım:")
        print("python predict_faz2.py ses.wav")

        sys.exit(1)

    predict(sys.argv[1])