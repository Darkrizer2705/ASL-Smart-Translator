import pandas as pd

df = pd.read_csv('datasets/gesture_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Phrases: {df["label"].nunique()}')
print(f'\nSamples per phrase:')
print(df['label'].value_counts().to_string())
