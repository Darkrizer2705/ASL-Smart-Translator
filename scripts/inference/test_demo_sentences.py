"""
Test demo sentences to verify if the ASL smart translator model is working.
Uses phrases that are available in the collect_phrases.py dataset.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.main import ASLTranslationPipeline


def test_demo_sentences():
    """Test the ASL translation pipeline with demo sentences."""
    
    # Initialize the pipeline
    print("Initializing ASL Translation Pipeline...")
    pipeline = ASLTranslationPipeline()
    
    # Demo sentences using phrases from the dataset
    demo_sentences = [
        "HELLO, WHAT IS YOUR NAME?",
        "WHERE IS THE BATHROOM? PLEASE HELP ME",
        "DO YOU UNDERSTAND? YES, I AM HAPPY"
    ]
    
    print("\n" + "="*70)
    print("ASL SMART TRANSLATOR - DEMO TEST")
    print("="*70)
    
    for idx, sentence in enumerate(demo_sentences, 1):
        print(f"\nTest {idx}: {sentence}")
        print("-" * 70)
        
        try:
            # Process the sentence through the pipeline
            result = pipeline.process(sentence)
            
            print(f"  Input Sentence: {sentence}")
            print(f"  Pipeline Output: {result}")
            print(f"  Status: ✓ SUCCESS")
            
        except Exception as e:
            print(f"  Input Sentence: {sentence}")
            print(f"  Error: {str(e)}")
            print(f"  Status: ✗ FAILED")
    
    print("\n" + "="*70)
    print("DEMO TEST COMPLETED")
    print("="*70)
    print("\nNote: Ensure you have:")
    print("  - Trained models in models/ directory")
    print("  - Collected phrase data in datasets/gesture_dataset.csv")
    print("  - Required dependencies installed")


if __name__ == "__main__":
    test_demo_sentences()
