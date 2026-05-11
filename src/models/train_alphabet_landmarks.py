# src/models/train_alphabet_landmarks.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("datasets/alphabet_landmarks.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

model = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=3,
                                random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

with open("models/alphabet_landmark_classifier.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)
print("🎉 Saved: models/alphabet_landmark_classifier.pkl")
