import numpy as np
import librosa
from scipy.stats import kurtosis, skew

SR = 22050

def extract_features(y, sr=SR):

    y = librosa.util.normalize(y)
    y = librosa.effects.preemphasis(y)

    features = []

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
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

    stft = np.abs(librosa.stft(y))

    centroid = librosa.feature.spectral_centroid(
        S=stft,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        S=stft,
        sr=sr
    )

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

    flatness = librosa.feature.spectral_flatness(
        S=stft
    )

    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    features.append(np.mean(rolloff85))
    features.append(np.std(rolloff85))

    features.append(np.mean(rolloff50))
    features.append(np.std(rolloff50))

    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    contrast = librosa.feature.spectral_contrast(
        S=stft,
        sr=sr
    )

    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

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

    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    

    features.append(np.mean(rms))
    features.append(np.std(rms))

    features.append(float(kurtosis(y)))
    features.append(float(skew(y)))

    return np.array(features, dtype=np.float32)