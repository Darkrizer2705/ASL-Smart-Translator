import mediapipe as mp
print(f"Version: {mp.__version__}")
print(f"Has solutions: {hasattr(mp, 'solutions')}")
print(f"Has tasks: {hasattr(mp, 'tasks')}")

# Try to import solutions
try:
    from mediapipe import solutions
    print("Solutions module found")
    print(f"Has hands: {hasattr(solutions, 'hands')}")
except Exception as e:
    print(f"Solutions module NOT found: {e}")

# Check what's in mp
attrs = [x for x in dir(mp) if not x.startswith('_')]
print(f"Available: {attrs}")
