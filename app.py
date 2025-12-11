from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import pickle
import os
from datetime import datetime
import re

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# ======================================================
# UNIVERSAL STUTTER CLEANER (ALL STUTTER TYPES)
# ======================================================
def clean_stutter(text: str):
    original_text = text
    highlighted_original = text
    cleaned = text

    # 1️⃣ Full word repetition
    pattern_word = r"\b(\w+)(?:\s+\1\b)+"
    while re.search(pattern_word, cleaned, flags=re.IGNORECASE):
        m = re.search(pattern_word, cleaned, flags=re.IGNORECASE)
        before = m.group(0)
        after = m.group(1)

        highlighted_original = highlighted_original.replace(
            before, f"<span style='color:red;font-weight:bold;'>{before}</span>"
        )
        cleaned = re.sub(pattern_word, after, cleaned, flags=re.IGNORECASE)

    # 2️⃣ Syllable repetition
    pattern_syllable = r"\b([A-Za-z]{1,4})-\1-([A-Za-z]+)\b"
    while re.search(pattern_syllable, cleaned):
        m = re.search(pattern_syllable, cleaned)
        before = m.group(0)
        after = m.group(1) + m.group(2)

        highlighted_original = highlighted_original.replace(
            before, f"<span style='color:red;font-weight:bold;'>{before}</span>"
        )
        cleaned = cleaned.replace(before, after)

    # 3️⃣ c-c-can type
    pattern_letter = r"\b([A-Za-z])(?:-\1)+([A-Za-z]+)\b"
    while re.search(pattern_letter, cleaned):
        m = re.search(pattern_letter, cleaned)
        before = m.group(0)
        after = m.group(1) + m.group(2)

        highlighted_original = highlighted_original.replace(
            before, f"<span style='color:red;font-weight:bold;'>{before}</span>"
        )
        cleaned = cleaned.replace(before, after)

    # 4️⃣ Leading repeated letters (THIS FIXES Sssssstop!)
    pattern_leading = r"\b([A-Za-z])\1{1,}([A-Za-z]+)\b"
    while re.search(pattern_leading, cleaned):
        m = re.search(pattern_leading, cleaned)
        before = m.group(0)
        after = m.group(1) + m.group(2)

        highlighted_original = highlighted_original.replace(
            before, f"<span style='color:red;font-weight:bold;'>{before}</span>"
        )
        cleaned = cleaned.replace(before, after)

    # 5️⃣ Mixed p-pl-please, w-w-w-want
    pattern_mix = r"\b([A-Za-z])(?:-[A-Za-z]){1,4}([A-Za-z]+)\b"
    while re.search(pattern_mix, cleaned):
        m = re.search(pattern_mix, cleaned)
        before = m.group(0)
        after = m.group(1) + m.group(2)

        highlighted_original = highlighted_original.replace(
            before, f"<span style='color:red;font-weight:bold;'>{before}</span>"
        )
        cleaned = cleaned.replace(before, after)

    highlighted_corrected = (
        f"<span style='color:#00ffb3;font-weight:bold;'>{cleaned}</span>"
    )

    stutter_found = (cleaned != original_text)

    return cleaned, highlighted_original, highlighted_corrected, stutter_found
# ======================================================
# FRONTEND ROUTE
# ======================================================
@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# AUDIO FEATURE EXTRACTION
# ======================================================
def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return feat
    except:
        return None


# ======================================================
# AUDIO PREDICTION API
# ======================================================
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file selected"})

    audio = request.files["audio"]

    os.makedirs("temp", exist_ok=True)
    filename = f"audio_{datetime.now().timestamp()}.wav"
    path = os.path.join("temp", filename)
    audio.save(path)

    features = extract_features(path)
    if features is None:
        return jsonify({"error": "Audio processing failed"})

    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]

    try:
        confidence = float(model.predict_proba(scaled)[0][1] * 100)
    except:
        confidence = None

    result = "Stutter Detected" if pred == 1 else "No Stutter"

    return jsonify({
        "result": result,
        "confidence": confidence,
        "file": audio.filename
    })


# ======================================================
# TEXT STUTTER ANALYSIS API
# ======================================================
@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.form.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"})

    cleaned, h_original, h_corrected, found = clean_stutter(text)

    result = "Stutter Detected in Text" if found else "No Stutter Found in Text"

    return jsonify({
        "result": result,
        "original": h_original,
        "corrected": h_corrected
    })


# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    app.run(debug=True, port=5050)
