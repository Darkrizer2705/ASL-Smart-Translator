"""
QUICK START: Fix Phrase Prediction

PROBLEM IDENTIFIED:
- Working phrases (8): AGAIN, FAMILY, HELP, HOW, ME, UNDERSTAND, WHAT, YES
  → These have HIGH variance in hand position (natural movement in training data)
  
- Broken phrases (17): All others
  → These have VERY LOW variance (too static, same hand position repeated)
  
The model was trained on static hand positions, so it can't recognize
real-time predictions where hands move naturally.

SOLUTION: Recollect data for broken phrases with natural hand movement
"""

import subprocess
import sys

def run_step(name, command):
    """Run a command and report results"""
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"{'='*70}")
    print(f"Command: {command}\n")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f"\n✅ {name} - SUCCESS")
    else:
        print(f"\n❌ {name} - FAILED")
    return result.returncode == 0

print("""
╔═══════════════════════════════════════════════════════════════════╗
║           PHRASE PREDICTION FIX - QUICK START GUIDE              ║
╚═══════════════════════════════════════════════════════════════════╝

ROOT CAUSE:
-----------
Your dataset has an inconsistency:
  • Working phrases (8): High variance in hand position
  • Broken phrases (17): Very low variance (too static)

The model learned from static positions but real-time predictions
have natural hand movement - that's why it fails.

STEP-BY-STEP FIX:
-----------------
""")

# Step 1: Recollect data
step1_success = run_step(
    "Recollect data for 17 broken phrases",
    "python recollect_broken_phrases.py"
)

if step1_success:
    # Step 2: Retrain
    step2_success = run_step(
        "Retrain model with new data",
        "python src/models/train_phrases.py"
    )
    
    if step2_success:
        # Step 3: Test
        print(f"\n{'='*70}")
        print("STEP: Test improved model")
        print(f"{'='*70}")
        print("""
Ready to test! Run this:
  python src/inference/predict_phrase.py

You should see much better phrase recognition for all 25 phrases!
        """)

print(f"\n{'='*70}")
print("KEY POINTS FOR DATA COLLECTION:")
print(f"{'='*70}")
print("""
When collecting data for the 17 broken phrases:

✓ DO THIS:
  - Move your hand around while signing
  - Change hand distance from camera
  - Rotate your hand slightly
  - Make natural movements
  - Collect 100-200 samples per phrase

✗ DON'T DO THIS:
  - Hold hand in exact same position
  - Keep hand perfectly still
  - Only use one hand position
  - Rush through the collection

This variation is CRITICAL for real-time recognition!
""")
