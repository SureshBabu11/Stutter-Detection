import os
import re
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

AUDIO_FOLDER = "Audio"

# ---------------------------------------------------------
# Build ClipId -> audio filename mapping
# ---------------------------------------------------------
def build_id_mapping():
    pattern = re.compile(r".*_(\d+)\.(wav|mp3|m4a)$", re.IGNORECASE)
    mapping = {}

    all_files = os.listdir(AUDIO_FOLDER)
    print("Total audio files found:", len(all_files))

    for f in all_files:
        match = pattern.match(f)
        if match:
            clip_id = int(match.group(1))
            mapping[clip_id] = os.path.join(AUDIO_FOLDER, f)

    print("Total mapped ClipIds:", len(mapping))

    # Debug: show first few mappings
    print("\nFirst 30 mappings:")
    for i, (cid, fname) in enumerate(list(mapping.items())[:30]):
        print(f"{i} -> {os.path.basename(fname)}")

    return mapping


# ---------------------------------------------------------
# Extract MFCC features
# ---------------------------------------------------------
def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        return feat
    except Exception as e:
        print("Error reading:", path, e)
        return None


# ---------------------------------------------------------
# Load labels from CSV
# ---------------------------------------------------------
def load_labels(csv_path):
    df = pd.read_csv(csv_path)

    stutter_cols = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
    df["stutter"] = df[stutter_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    return df["ClipId"].astype(int), df["stutter"].astype(int)


# ---------------------------------------------------------
# MAIN TRAINING PROCESS
# ---------------------------------------------------------
def main():
    clip_ids, labels = load_labels("SEP-28k_labels.csv")

    audio_map = build_id_mapping()

    X = []
    y = []

    print("\nExtracting features...")

    for clip_id, label in zip(clip_ids, labels):
        if clip_id not in audio_map:
            continue

        audio_path = audio_map[clip_id]
        feat = extract_features(audio_path)

        if feat is not None:
            X.append(feat)
            y.append(label)

    if len(X) == 0:
        print("ERROR: No usable audio found. Check mapping / filenames.")
        return

    X = np.array(X)
    y = np.array(y)

    print("\nTotal usable training samples:", len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    print("\nSaved model.pkl and scaler.pkl successfully!")


if __name__ == "__main__":
    main()
