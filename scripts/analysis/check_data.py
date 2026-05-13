import pandas as pd
import pickle

df = pd.read_csv('datasets/gesture_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Unique labels: {df["label"].nunique()}')
print(f'Classes: {list(df["label"].unique())}')
print(f'\nSamples per class:')
print(df["label"].value_counts().sort_index())
print(f'\nColumns: {list(df.columns)}')
print(f'Shape: {df.shape}')

# Check model
print("\n\n=== MODEL INFO ===")
try:
    with open('models/phrase_classifier.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        encoder = data['encoder']
        print(f"Model features expected: {model.n_features_in_}")
        print(f"Encoder classes: {list(encoder.classes_)}")
        print(f"Number of classes: {len(encoder.classes_)}")
except Exception as e:
    print(f"Error loading model: {e}")