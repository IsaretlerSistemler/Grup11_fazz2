"""
============================================================
 EMO-CHALLENGE 2026 — UPGRADED CLEAN VERSION
============================================================
"""

import os
import numpy as np
import librosa
import pickle
import warnings
from feature_extractor import extract_features
from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATASET_PATH = "data"

SR = 22050


# ============================================================
# AUGMENTATION (SAFE VERSION)
# ============================================================
def augment_audio(y, sr):
    augmented = []

    # noise (low level)
    noise = y + 0.002 * np.random.randn(len(y))
    augmented.append(noise)

    # pitch shift (safe range)
    pitch = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1)
    augmented.append(pitch)

    return augmented


# =============

# ============================================================
# LOAD SINGLE FILE
# ============================================================
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)

        if len(y) < SR * 0.2:
            return None

        return y, sr

    except:
        return None


# ============================================================
# DATASET LOADING (FIXED LEAKAGE)
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

            audio = load_audio(path)
            if audio is None:
                continue

            y_audio, sr = audio

            # original
            feat = extract_features(y_audio, sr)
            X.append(feat)
            y.append(label)

            # augmentation (ONLY TRAIN LATER)
            for aug in augment_audio(y_audio, sr):
                feat_aug = extract_features(aug, sr)
                X.append(feat_aug)
                y.append(label)

    return np.array(X), np.array(y)


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":

    X, y_raw = load_dataset(DATASET_PATH)

    print(f"\n📊 Total samples: {len(X)}")
    print(f"📐 Feature size: {X.shape[1]}")

    # LABEL ENCODER
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # SPLIT (BEFORE SCALING & SELECTION)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # SCALING
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # FEATURE SELECTION (more stable k)
    selector = SelectKBest(f_classif, k=60)

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print(f"\n✅ Final feature size: {X_train.shape[1]}")

    # MODELS
    svm = SVC(
        C=15,
        gamma="scale",
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\n🧠 Training models...")

    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # ENSEMBLE (BALANCED)
    ensemble = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("rf", rf)
        ],
        voting="soft",
        weights=[3, 3]
    )

    print("\n🤝 Training ensemble...")

    ensemble.fit(X_train, y_train)

    # PREDICTION
    y_pred = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"🎯 ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 60)

    print("\n📊 Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # SAVE MODEL
    model_package = {
    "model": ensemble,
    "scaler": scaler,
    "selector": selector,
    "label_encoder": le,
    "X_test": X_test,
    "y_test": y_test
}
    with open("emotion_model_faz2.pkl", "wb") as f:
        pickle.dump(model_package, f)

    print("\n💾 Model saved: emotion_model_faz2.pkl")