import os
import shutil
import unicodedata

source = "dataset"
target = "data"

# ---------------------------
# CLEAN FUNCTION (GÜÇLENDİRİLDİ)
# ---------------------------
def clean(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    
    text = text.lower()
    text = text.replace(" ", "")
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = text.replace(".", "")
    
    return text

# ---------------------------
# EMOTION DETECTION (FINAL)
# ---------------------------
def detect_emotion(file):

    f = clean(file)

    # HAPPY
    if any(x in f for x in ["mutlu", "happy"]):
        return "happy"

    # SAD
    if any(x in f for x in ["uzgun", "sad", "mutsuz"]):
        return "sad"

    # ANGRY
    if any(x in f for x in ["ofkeli", "angry", "furious", "fury", "kizgin"]):
        return "angry"

    # NEUTRAL
    if any(x in f for x in ["notr", "neutral"]):
        return "neutral"

    # SURPRISED (GÜÇLENDİRİLDİ)
    if any(x in f for x in [
        "saskin", "saskin", "sasirma", "saskirma",
        "surprised", "surprise", "shock", "sasir", "sasiran"
    ]):
        return "surprised"

    return None  # unknown tamamen kaldırıldı

# ---------------------------
# FOLDERS
# ---------------------------
labels = ["happy", "sad", "angry", "neutral", "surprised"]

for l in labels:
    os.makedirs(os.path.join(target, l), exist_ok=True)

# ---------------------------
# PROCESS
# ---------------------------
count = 0

for root, dirs, files in os.walk(source):
    for file in files:
        if file.lower().endswith(".wav"):

            label = detect_emotion(file)

            # UNKNOWN DOSYALARI ATLA
            if label is None:
                continue

            src = os.path.join(root, file)
            dst = os.path.join(target, label, file)

            shutil.copy2(src, dst)
            count += 1

# ---------------------------
# RESULT
# ---------------------------
print("\n🚀 Dataset temizleme tamamlandı")
print("Toplam işlenen dosya:", count)


for c in os.listdir("data"):
    print(c, len(os.listdir(os.path.join("data", c))))