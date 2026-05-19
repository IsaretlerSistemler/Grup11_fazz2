import sys
import pickle
import numpy as np
import librosa
import warnings

from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path, sr_target=22050, n_mfcc=40):

    try:

        y, sr = librosa.load(
            file_path,
            sr=sr_target,
            mono=True
        )

        if len(y) < sr * 0.1:
            return None

        y = librosa.util.normalize(y)

        y = librosa.effects.preemphasis(y)

        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc
        )

        mfcc_delta = librosa.feature.delta(mfcc)

        mfcc_delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))

        features.extend(np.mean(mfcc[:13], axis=1))
        features.extend(np.std(mfcc[:13], axis=1))

        # STFT
        stft = np.abs(librosa.stft(y))

        centroid = librosa.feature.spectral_centroid(
            S=stft,
            sr=sr
        )

        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(
            S=stft,
            sr=sr
        )

        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))

        rolloff85 = librosa.feature.spectral_rolloff(
            S=stft,
            sr=sr,
            roll_percent=0.85
        )

        rolloff50 = librosa.feature.spectral_rolloff(
            S=stft,
            sr=sr,
            roll_percent=0.50
        )

        features.append(np.mean(rolloff85))
        features.append(np.std(rolloff85))

        features.append(np.mean(rolloff50))
        features.append(np.std(rolloff50))

        flatness = librosa.feature.spectral_flatness(
            S=stft
        )

        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        contrast = librosa.feature.spectral_contrast(
            S=stft,
            sr=sr
        )

        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        # MEL
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        features.append(np.mean(mel_db))
        features.append(np.std(mel_db))

        features.append(np.min(mel_db))
        features.append(np.max(mel_db))

        features.append(np.mean(mel_db[:40]))
        features.append(np.mean(mel_db[40:80]))
        features.append(np.mean(mel_db[80:]))

        # TIME DOMAIN
        zcr = librosa.feature.zero_crossing_rate(y)

        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rms = librosa.feature.rms(y=y)

        features.append(np.mean(rms))
        features.append(np.std(rms))

        # STATISTICS
        features.append(float(kurtosis(y)))
        features.append(float(skew(y)))

        return np.array(features, dtype=np.float32)

    except Exception as e:

        print(f"[HATA] {e}")

        return None


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

    features = extract_features(audio_path)

    if features is None:

        print("❌ Feature çıkarılamadı")

        return

    # scale
    features_sc = scaler.transform(
        features.reshape(1, -1)
    )

    # select
    features_sc = selector.transform(features_sc)

    # predict
    pred_idx = model.predict(features_sc)[0]

    pred_label = le.inverse_transform([pred_idx])[0]

    # probabilities
    probas = model.predict_proba(features_sc)[0]

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