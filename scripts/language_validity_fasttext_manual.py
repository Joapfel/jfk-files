import json
import random
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
FASTTEXT_RESULTS_FILE = DATA_DIR / 'languageValidity_fasttext.json'
MANUAL_VALIDATION_FILE = DATA_DIR / 'validity/manual_fasttext_validation.json'
VALIDITY_DIR = DATA_DIR / 'validity'

def load_fasttext_results():
    """Load the FastText language validity results."""
    with open(FASTTEXT_RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_manual_validation():
    """Load existing manual validations if they exist."""
    if MANUAL_VALIDATION_FILE.exists():
        with open(MANUAL_VALIDATION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_manual_validation(validations):
    """Save manual validations to file."""
    VALIDITY_DIR.mkdir(exist_ok=True)
    with open(MANUAL_VALIDATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(validations, f, indent=2, ensure_ascii=False)

def get_file_content(rel_path):
    """Get the content of a file given its relative path."""
    file_path = DATA_DIR / rel_path
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def get_user_input(prompt, valid_choices=None):
    """Get and validate user input."""
    while True:
        user_input = input(prompt).strip().lower()
        if valid_choices is None or user_input in valid_choices:
            return user_input
        print(f"Please enter one of: {', '.join(valid_choices)}")

def display_file_info(file_path, result):
    """Display file information and content."""
    print("\n" + "="*80)
    print(f"File: {file_path}")
    print(f"Detected Language: {result.get('language', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Auto-Valid: {result.get('valid', False)}")
    print(f"Reason: {result.get('reason', 'No reason provided')}")
    print("-"*80)
    
    content = get_file_content(file_path)
    print(content[:1000] + ("..." if len(content) > 1000 else ""))
    print("="*80)

def main():
    # Load data
    results = load_fasttext_results()
    validations = load_manual_validation()
    
    # Get list of files to validate
    files_to_validate = [f for f in results.keys() if f not in validations]
    
    if not files_to_validate:
        print("All files have been manually validated!")
        return
    
    print(f"Found {len(files_to_validate)} files to validate. Starting validation...")
    
    for file_path in files_to_validate:
        result = results[file_path]
        display_file_info(file_path, result)
        
        # Get user input
        action = get_user_input(
            "Is this valid human language? (y)es, (n)o, (s)kip, (q)uit: ",
            ['y', 'n', 's', 'q']
        )
        
        if action == 'q':
            print("\nSaving progress and quitting...")
            break
        elif action == 's':
            print("Skipping this file...\n")
            continue
        
        # Get additional notes
        notes = input("Any notes about this validation (press Enter for none): ").strip()
        
        # Save validation
        validations[file_path] = {
            'auto_valid': result.get('valid', False),
            'manual_valid': (action == 'y'),
            'detected_language': result.get('language'),
            'confidence': result.get('confidence'),
            'notes': notes,
            'auto_reason': result.get('reason', '')
        }
        
        # Save after each validation
        save_manual_validation(validations)
        print(f"Saved validation for {file_path}\n")
    
    # Final save
    save_manual_validation(validations)
    print(f"\nValidation complete! Validated {len(validations)} files.")
    print(f"Results saved to: {MANUAL_VALIDATION_FILE}")

if __name__ == "__main__":
    main()