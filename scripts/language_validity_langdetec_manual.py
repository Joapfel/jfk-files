import json
import random
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_FILE = DATA_DIR / 'validity/languageValidity-langdetect.json'
VALIDITY_DIR = DATA_DIR / 'validity'

def load_results():
    """Load the language validity results."""
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_content(rel_path):
    """Get the content of a file given its relative path."""
    file_path = DATA_DIR / rel_path
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_user_input():
    """Get and validate user input (y/n)."""
    while True:
        user_input = input("Is this valid human language? (y/n/q to quit): ").strip().lower()
        if user_input in ['y', 'n', 'q']:
            return user_input
        print("Please enter 'y', 'n', or 'q' to quit.")

def main():
    # Load results
    results = load_results()
    
    # Separate valid and invalid files
    valid_files = [f for f, r in results.items() if r['valid']]
    invalid_files = [f for f, r in results.items() if not r['valid']]
    
    # Sample 25 from each group
    sample_size = 25
    sample = (
        random.sample(valid_files, min(sample_size, len(valid_files))) +
        random.sample(invalid_files, min(sample_size, len(invalid_files)))
    )
    random.shuffle(sample)
    
    total = len(sample)
    correct = 0
    
    print(f"\n{'='*80}\nLanguage Validation Check\n{'='*80}")
    print(f"Reviewing {total} files (mix of valid and invalid)\n")
    
    for i, rel_path in enumerate(sample, 1):
        result = results[rel_path]
        content = get_file_content(rel_path)
        
        print(f"\n{'='*80}")
        print(f"File {i}/{total}: {rel_path}")
        print(f"Model prediction: {'VALID' if result['valid'] else 'INVALID'}")
        if result['language']:
            print(f"Detected language: {result['language']} (confidence: {result['confidence']:.2f})")
        if result.get('reason'):
            print(f"Reason: {result['reason']}")
        print("-"*40)
        print(f"Content (first 1000 chars):\n{content[:1000]}...")
        print("-"*40)
        
        user_input = get_user_input()
        if user_input == 'q':
            print("\nValidation aborted by user.")
            return
            
        user_valid = (user_input == 'y')
        if user_valid == result['valid']:
            correct += 1
            print("✓ Correct!")
        else:
            print("✗ Incorrect")
        
        accuracy = (correct / i) * 100
        print(f"\nCurrent accuracy: {accuracy:.1f}% ({correct}/{i})")
    
    print("\n" + "="*80)
    print("Validation complete!")
    print(f"Final accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Save results
    os.makedirs(VALIDITY_DIR, exist_ok=True)
    with open(VALIDITY_DIR / 'manual_validation_results.txt', 'w') as f:
        f.write(f"Manual Validation Results\n{'='*25}\n")
        f.write(f"Files reviewed: {total}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.1f}%\n\n")
        f.write("Model was correct on:\n")
        for rel_path in sample:
            f.write(f"- {rel_path}: {'✓' if results[rel_path]['valid'] == (results[rel_path]['valid'] == 'y') else '✗'}\n")

if __name__ == "__main__":
    main()