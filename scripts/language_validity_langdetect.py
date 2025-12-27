import os
import json
from pathlib import Path
from langdetect import detect_langs, LangDetectException

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_FILE = DATA_DIR / 'languageValidity.json'

# Part directories to process
PART_DIRS = [
    'Part 1',
    'Part 2',
    'Part 3'
]

# Language settings
VALID_LANGUAGES = {'en'}  # English
MIN_CONFIDENCE = 0.8
MIN_LENGTH = 20

def is_valid_language(text: str) -> dict:
    """Check if text contains valid human language using langdetect."""
    text = text.strip()
    
    # Check minimum length
    if len(text) < MIN_LENGTH:
        return {
            "valid": False,
            "reason": f"Text too short (min {MIN_LENGTH} chars required)",
            "language": None,
            "confidence": 0.0
        }
    
    try:
        # Get language prediction
        result = detect_langs(text)
        language_code = result[0].lang
        confidence = result[0].prob
        
        return {
            "valid": language_code in VALID_LANGUAGES and confidence >= MIN_CONFIDENCE,
            "language": language_code,
            "confidence": confidence,
            "reason": "" if language_code in VALID_LANGUAGES else f"Unsupported language: {language_code}"
        }
    except LangDetectException as e:
        return {
            "valid": False,
            "reason": f"Language detection error: {str(e)}",
            "language": None,
            "confidence": 0.0
        }

def process_files(max_files: int = None) -> dict:
    """Process text files in the part directories.
    
    Args:
        max_files: Maximum number of files to process (None for all)
    """
    results = {}
    files_processed = 0
    
    for part in PART_DIRS:
        part_path = DATA_DIR / part / 'txt'
        if not part_path.exists():
            print(f"Warning: Directory not found: {part_path}")
            continue
            
        print(f"\nProcessing files in: {part}")
        
        for txt_file in part_path.glob('*.txt'):
            if max_files is not None and files_processed >= max_files:
                print(f"\nReached maximum of {max_files} files. Stopping processing.")
                return results
                
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Get relative path for the output
                rel_path = str(txt_file.relative_to(DATA_DIR)).replace('\\', '/')
                
                # Check language validity
                results[rel_path] = is_valid_language(content)
                files_processed += 1
                
                print(f"  {txt_file.name}: "
                      f"Valid: {results[rel_path]['valid']} "
                      f"({results[rel_path]['language']}, {results[rel_path]['confidence']:.2f})")
                        
            except Exception as e:
                print(f"Error processing {txt_file}: {str(e)}")
                results[str(txt_file)] = {
                    "valid": False,
                    "reason": f"Error: {str(e)}",
                    "language": None,
                    "confidence": 0.0
                }
    
    return results

def save_results(results: dict):
    """Save results to the output JSON file."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

def main():
    print(f"Starting language validation for JFK files in: {DATA_DIR}")
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process files (set max_files=None to process all files)
    results = process_files(max_files=None)
    save_results(results)
    
    # Print summary
    valid_count = sum(1 for r in results.values() if r["valid"])
    print("\nSummary:")
    print(f"Total files processed: {len(results)}")
    print(f"Valid language files: {valid_count} ({(valid_count/len(results)*100 if results else 0):.1f}%)")
    print(f"Invalid/Unreadable files: {len(results) - valid_count}")

if __name__ == "__main__":
    main()
