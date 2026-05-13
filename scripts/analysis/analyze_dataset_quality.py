"""
Deep analysis of dataset quality and feature consistency
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("DATASET QUALITY ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('datasets/gesture_dataset.csv')

# Working phrases (can predict correctly)
WORKING = ['AGAIN', 'FAMILY', 'HELP', 'HOW', 'ME', 'UNDERSTAND', 'WHAT', 'YES']
NOT_WORKING = [p for p in df['label'].unique() if p not in WORKING]

print(f"\n✓ Working phrases ({len(WORKING)}): {WORKING}")
print(f"✗ Not working ({len(NOT_WORKING)}): {NOT_WORKING}")

# Check data distribution
print(f"\n📊 SAMPLES PER PHRASE:")
for phrase in sorted(df['label'].unique()):
    count = (df['label'] == phrase).sum()
    status = "✓" if phrase in WORKING else "✗"
    print(f"  {status} {phrase:15} : {count:4} samples")

# Check feature quality - look for NaN, inf, or constant values
print(f"\n🔍 DATA QUALITY CHECKS:")
feature_cols = [c for c in df.columns if c != 'label']

print(f"\nNaN values: {df[feature_cols].isna().sum().sum()}")
print(f"Inf values: {np.isinf(df[feature_cols]).sum().sum()}")

# Check feature statistics
print(f"\nFeature ranges (all data):")
print(f"  Min: {df[feature_cols].min().min():.4f}")
print(f"  Max: {df[feature_cols].max().max():.4f}")
print(f"  Mean: {df[feature_cols].mean().mean():.4f}")
print(f"  Std: {df[feature_cols].std().mean():.4f}")

# Check if features are reasonable (landmarks should be 0-1 range after normalization)
out_of_range = ((df[feature_cols] < -1) | (df[feature_cols] > 2)).sum().sum()
print(f"\nValues outside [-1, 2] range: {out_of_range} (should be ~0)")

# Feature variance per phrase
print(f"\n📈 FEATURE VARIANCE BY PHRASE:")
print(f"\n{'Phrase':15} | Mean Var | Std Var | Min Var | Max Var")
print(f"{'-'*70}")

for phrase in sorted(df['label'].unique()):
    phrase_df = df[df['label'] == phrase][feature_cols]
    variances = phrase_df.var()
    status = "✓" if phrase in WORKING else "✗"
    print(f"{status} {phrase:13} | {variances.mean():8.4f} | {variances.std():7.4f} | {variances.min():7.4f} | {variances.max():7.4f}")

# Check for data collection issues - look at feature distributions
print(f"\n🔎 CHECKING FEATURE CONSISTENCY:")

# Get feature statistics for working vs not working
working_data = df[df['label'].isin(WORKING)][feature_cols]
not_working_data = df[df['label'].isin(NOT_WORKING)][feature_cols]

print(f"\nWorking phrases - feature stats:")
print(f"  Mean: {working_data.mean().mean():.4f}, Std: {working_data.std().mean():.4f}")
print(f"  Range: [{working_data.min().min():.4f}, {working_data.max().max():.4f}]")

print(f"\nNot working - feature stats:")
print(f"  Mean: {not_working_data.mean().mean():.4f}, Std: {not_working_data.std().mean():.4f}")
print(f"  Range: [{not_working_data.min().min():.4f}, {not_working_data.max().max():.4f}]")

# Check for constant features
print(f"\n⚠️  CHECKING FOR ZERO-VARIANCE FEATURES:")
constant_features = [c for c in feature_cols if df[c].std() < 0.001]
if constant_features:
    print(f"  Found {len(constant_features)} constant features: {constant_features[:5]}...")
else:
    print(f"  ✓ No constant features found")

# Check frame distribution - are all samples independent or duplicates?
print(f"\n📹 CHECKING FOR DUPLICATE SAMPLES:")
duplicate_rows = df[feature_cols].duplicated().sum()
print(f"  Exact duplicate rows: {duplicate_rows}")

# Check if landmark extraction might be the issue
print(f"\n🚨 CHECKING LANDMARK EXTRACTION ISSUE:")
print(f"\nExpected features: 63 (21 landmarks × 3 coords)")
print(f"Actual features: {len(feature_cols)}")

# Check if the data looks normalized (wrist-relative)
# If properly normalized, most values should be close to 0 (wrist at origin)
mean_values = working_data.mean()
print(f"\nMean feature values (working phrases):")
print(f"  x0, y0, z0 (should be ~0): {mean_values[['x0', 'y0', 'z0']].values}")
print(f"  Other landmarks: mean ~{mean_values[[c for c in feature_cols if c not in ['x0', 'y0', 'z0']]].mean():.4f}")

# Hypothesis: Check if API version matters
print(f"\n💡 HYPOTHESIS CHECK:")
print(f"If only 8/25 phrases work, likely causes:")
print(f"  1. API mismatch: Training data from OLD API, prediction uses NEW API")
print(f"  2. Inconsistent normalization: Different wrist point used")
print(f"  3. Data collection issue: Some phrases collected differently")
print(f"  4. Frame skipping: Training used every frame, real-time skips frames")

print("\n" + "=" * 80)