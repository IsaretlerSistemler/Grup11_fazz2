"""
============================================================
 EMO-CHALLENGE 2026 — FAZ 2 OPTIMIZED VERSION
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
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

DATASET_PATH = "data"


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

        # normalize
        y = librosa.util.normalize(y)

        # preemphasis
        y = librosa.effects.preemphasis(y)

        features = []

        # ====================================================
        # MFCC
        # ====================================================

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

        # ilk 13 mfcc
        features.extend(np.mean(mfcc[:13], axis=1))
        features.extend(np.std(mfcc[:13], axis=1))

        # ====================================================
        # STFT
        # ====================================================

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

        # ====================================================
        # MEL SPECTROGRAM
        # ====================================================

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

        # ====================================================
        # TIME DOMAIN
        # ====================================================

        zcr = librosa.feature.zero_crossing_rate(y)

        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rms = librosa.feature.rms(y=y)

        features.append(np.mean(rms))
        features.append(np.std(rms))

        # ====================================================
        # STATISTICS
        # ====================================================

        features.append(float(kurtosis(y)))
        features.append(float(skew(y)))

        return np.array(features, dtype=np.float32)

    except Exception as e:

        print(f"[HATA] {file_path}: {e}")

        return None


# ============================================================
# DATASET LOAD
# ============================================================

def load_dataset(dataset_path):

    X = []
    y = []

    print("\n🎧 Dataset yükleniyor...\n")

    for label in sorted(os.listdir(dataset_path)):

        class_path = os.path.join(
            dataset_path,
            label
        )

        if not os.path.isdir(class_path):
            continue

        files = [
            f for f in os.listdir(class_path)
            if f.endswith(".wav")
        ]

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

    # ========================================================
    # FEATURE CACHE
    # ========================================================

    if os.path.exists("X_features.npy"):

        print("\n⚡ Feature cache bulundu!")

        X = np.load("X_features.npy")
        y_raw = np.load("y_labels.npy")

    else:

        X, y_raw = load_dataset(DATASET_PATH)

        np.save("X_features.npy", X)
        np.save("y_labels.npy", y_raw)

        print("\n💾 Feature cache oluşturuldu!")

    print(f"\n📊 Toplam örnek : {len(X)}")
    print(f"📐 Feature sayısı : {X.shape[1]}")

    # ========================================================
    # LABEL ENCODER
    # ========================================================

    le = LabelEncoder()

    y = le.fit_transform(y_raw)

    # ========================================================
    # SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # SCALE
    # ========================================================

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    selector = SelectKBest(
        score_func=f_classif,
        k=120
    )

    X_train_sc = selector.fit_transform(
        X_train_sc,
        y_train
    )

    X_test_sc = selector.transform(X_test_sc)

    print(f"\n✅ Yeni feature boyutu : {X_train_sc.shape[1]}")

    # ========================================================
    # MODELS
    # ========================================================

    print("\n🧠 Modeller eğitiliyor...\n")

    svm = SVC(
        C=10,
        gamma='scale',
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    svm.fit(X_train_sc, y_train)

    gbm = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        random_state=42
    )

    gbm.fit(X_train_sc, y_train)

    # ========================================================
    # ENSEMBLE
    # ========================================================

    ensemble = VotingClassifier(
        estimators=[
            ("svm", svm),
            ("gbm", gbm)
        ],
        voting="soft",
        weights=[3, 2]
    )

    print("\n🤝 Ensemble eğitiliyor...\n")

    ensemble.fit(X_train_sc, y_train)

    # ========================================================
    # TEST
    # ========================================================

    y_pred = ensemble.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print(f"🎯 TEST ACCURACY : {acc:.4f} ({acc*100:.2f}%)")
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
        "model": ensemble,
        "scaler": scaler,
        "selector": selector,
        "label_encoder": le,
        "feature_dim": X.shape[1]
    }

    with open("emotion_model_faz2.pkl", "wb") as f:

        pickle.dump(save_dict, f)

    print("\n💾 Model kaydedildi : emotion_model_faz2.pkl")