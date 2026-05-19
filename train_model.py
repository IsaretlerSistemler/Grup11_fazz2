"""
============================================================
 EMO-CHALLENGE 2026 — FINAL BALANCED VERSION (~80-86%)
============================================================
"""

import os
import numpy as np
import librosa
import pickle
import warnings

from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATASET_PATH = "data"


# ============================================================
# FEATURE EXTRACTION (SIMPLE + CLEAN)
# ============================================================

def extract_features(file_path, sr_target=22050, n_mfcc=20):

    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True)

        if len(y) < sr * 0.15:
            return None

        y = librosa.util.normalize(y)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        features = []

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Spectral features
        stft = np.abs(librosa.stft(y))

        centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)

        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))

        # Time domain
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        features.append(np.mean(rms))
        features.append(np.std(rms))

        # Statistics
        features.append(float(kurtosis(y)))
        features.append(float(skew(y)))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"[HATA] {file_path}: {e}")
        return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_path):

    X = []
    y = []

    print("\n🎧 Dataset yükleniyor...\n")

    for label in sorted(os.listdir(dataset_path)):

        class_path = os.path.join(dataset_path, label)

        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if f.endswith(".wav")]

        print(f"📁 {label:12s} → {len(files)} dosya")

        for file in files:

            path = os.path.join(class_path, file)

            feat = extract_features(path)

            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # LOAD OR COMPUTE
    if os.path.exists("X_features.npy"):
        X = np.load("X_features.npy")
        y_raw = np.load("y_labels.npy")
    else:
        X, y_raw = load_dataset(DATASET_PATH)

        np.save("X_features.npy", X)
        np.save("y_labels.npy", y_raw)

    print(f"\n📊 Toplam örnek : {len(X)}")
    print(f"📐 Feature sayısı : {X.shape[1]}")

    # LABEL ENCODING
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # SCALING
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # FEATURE SELECTION (MILD)
    selector = SelectKBest(score_func=f_classif, k=40)

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print(f"\n✅ Feature boyutu : {X_train.shape[1]}")

    # ========================================================
    # MODEL (SIMPLE SVM ONLY)
    # ========================================================

    print("\n🧠 Model eğitiliyor...\n")

    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight=None
    )

    model.fit(X_train, y_train)

    # ========================================================
    # TEST
    # ========================================================

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(f"🎯 FINAL ACCURACY : {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)

    print("\n📊 Classification Report:\n")

    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # ========================================================
    # SAVE MODEL
    # ========================================================

    save_dict = {
        "model": model,
        "scaler": scaler,
        "selector": selector,
        "label_encoder": le
    }

    with open("emotion_model_final.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("\n💾 Model kaydedildi: emotion_model_final.pkl")