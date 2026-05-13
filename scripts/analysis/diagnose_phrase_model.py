import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

print("=" * 70)
print("PHRASE MODEL DIAGNOSTIC REPORT")
print("=" * 70)

# Load data
df = pd.read_csv('datasets/gesture_dataset.csv')
print(f"\n✓ Dataset loaded: {len(df)} samples, {df['label'].nunique()} phrases")

# Split data
X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {len(X_train)} | Test set: {len(X_test)}")

# Train model
print("\n⏳ Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Encode labels
le = LabelEncoder()
le.fit(y)

# Test predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 MODEL ACCURACY")
print(f"   Test Accuracy: {accuracy * 100:.2f}%")

# Detailed per-class accuracy
print(f"\n📈 PER-CLASS ACCURACY")
for phrase in sorted(np.unique(y)):
    mask = y_test == phrase
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], y_pred[mask])
        count = mask.sum()
        print(f"   {phrase:15} - {class_acc*100:5.1f}% ({count} samples)")

# Look for confused classes
print(f"\n⚠️  PROBLEMATIC PREDICTIONS")
conf_matrix = confusion_matrix(y_test, y_pred, labels=le.classes_)

# Find high confusion pairs
wrong_indices = np.where(y_pred != y_test)[0]
if len(wrong_indices) > 0:
    print(f"   Total misclassifications: {len(wrong_indices)} out of {len(y_test)}")
    
    # Show most common confusion pairs
    confusion_pairs = {}
    for idx in wrong_indices[:50]:  # Look at first 50 errors
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        conf = y_pred_proba[idx][le.transform([pred_label])[0]]
        key = (true_label, pred_label)
        if key not in confusion_pairs:
            confusion_pairs[key] = []
        confusion_pairs[key].append(conf)
    
    print("\n   Top confusion pairs (true → predicted):")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for (true_label, pred_label), confs in sorted_pairs:
        count = len(confs)
        avg_conf = np.mean(confs)
        print(f"      {true_label:15} → {pred_label:15} ({count}x, avg conf: {avg_conf:.1%})")

# Check for low confidence predictions
print(f"\n🔍 LOW CONFIDENCE PREDICTIONS")
max_probs = np.max(y_pred_proba, axis=1)
low_conf_mask = max_probs < 0.60
if low_conf_mask.sum() > 0:
    low_conf_count = low_conf_mask.sum()
    print(f"   Predictions with <60% confidence: {low_conf_count} ({low_conf_count/len(y_test)*100:.1f}%)")
    
    low_conf_accuracy = accuracy_score(y_test[low_conf_mask], y_pred[low_conf_mask])
    print(f"   Accuracy on low-confidence predictions: {low_conf_accuracy*100:.1f}%")
else:
    print(f"   All predictions have ≥60% confidence ✓")

# Check for balanced prediction distribution
print(f"\n⚖️  PREDICTION DISTRIBUTION")
unique, counts = np.unique(y_pred, return_counts=True)
for phrase, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]:
    print(f"   {phrase:15} predicted {count:4} times ({count/len(y_pred)*100:5.1f}%)")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)

if accuracy < 0.85:
    print("❌ Model accuracy is below 85%. Consider:")
    print("   - Collect more diverse training data")
    print("   - Check data quality (hand detection consistency)")
    print("   - Try different model hyperparameters")
    print("   - Verify landmark extraction is working correctly")
else:
    print("✓ Model accuracy is good (>85%)")

if len(wrong_indices) > len(y_test) * 0.15:
    print("\n❌ Many misclassifications. Debug the confused pairs:")
    for (true_label, pred_label), confs in sorted_pairs[:3]:
        print(f"   - {true_label} vs {pred_label}: May need more distinct training data")

# Save model
print("\n💾 Saving retrained model...")
with open('models/phrase_classifier.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoder': le}, f)

print("✓ Model saved!")