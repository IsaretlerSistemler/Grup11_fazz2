import os
import numpy as np
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = os.path.join("dataset", "Dataset")


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)

        # 🧠 1. MFCC (çok önemli)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 🔊 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # 🔥 3. RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # 🎵 4. Spectral Centroid (DUYGU için çok önemli)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        # 🎼 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)

        # ❌ Pitch (KALDIRILDI → hatalı ve noise yapıyor)

        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            zcr_mean,
            zcr_std,
            rms_mean,
            rms_std,
            centroid_mean,
            centroid_std,
            bandwidth_mean
        ])

        return features

    except Exception as e:
        print("Hata:", file_path, e)
        return None


X = []
y = []

print("🎧 Dataset yükleniyor...")

for label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_path):
        continue

    for file in os.listdir(class_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(class_path, file)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"📊 Toplam sample: {len(X)}")

# 🔥 Label encode (çok kritik iyileştirme)
le = LabelEncoder()
y = le.fit_transform(y)

# 🔥 Güvenli split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n🧠 Model eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", acc)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 💾 model + label encoder kaydet
with open("emotion_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "label_encoder": le
    }, f)

print("\n💾 Model kaydedildi: emotion_model.pkl")