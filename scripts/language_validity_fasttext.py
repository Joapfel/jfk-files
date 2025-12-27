import os
import json
from pathlib import Path
import fasttext
import urllib.request

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_PATH = BASE_DIR / 'lid.176.bin'
OUTPUT_FILE = DATA_DIR / 'languageValidity_fasttext.json'

# Part directories to process
PART_DIRS = [
    'Part 1',
    'Part 2',
    'Part 3'
]

# Language settings
VALID_LANGUAGES = {'en'}  # FastText uses __label__ prefix
MIN_CONFIDENCE = 0.8
MIN_LENGTH = 20

def download_model():
    """Download the FastText language identification model if not present."""
    if not MODEL_PATH.exists():
        print("Downloading FastText language identification model...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            str(MODEL_PATH)
        )
        print("Download complete!")

def load_model():
    """Load the FastText model."""
    download_model()
    return fasttext.load_model(str(MODEL_PATH))

def is_valid_language(text: str, model) -> dict:
    """Check if text contains valid human language using FastText."""
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
        # Replace newlines with spaces and process as a single text
        text = ' '.join(text.split('\n'))
        
        try:
            # Use a more robust way to get predictions
            labels, scores = model.predict(text, k=1)
            best_lang = labels[0].replace('__label__', '')
            best_confidence = float(scores[0])

        except Exception as e:
            # Add more detailed error information
            import traceback
            error_details = f"{str(e)}\n\n{traceback.format_exc()}"
            return {
                "valid": False,
                "reason": f"Language detection error: {error_details}",
                "language": None,
                "confidence": 0.0
            }
            
        return {
            "valid": best_lang in VALID_LANGUAGES and best_confidence >= MIN_CONFIDENCE,
            "language": best_lang.replace('__label__', ''),
            "confidence": best_confidence,
            "reason": "" if best_lang in VALID_LANGUAGES else f"Unsupported language: {best_lang}"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Language detection error: {str(e)}",
            "language": None,
            "confidence": 0.0
        }

def process_files(max_files: int = None) -> dict:
    """Process text files in the part directories."""
    print("Loading FastText model...")
    model = load_model()
    print("Model loaded successfully!")
    
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
                results[rel_path] = is_valid_language(content, model)
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
    results = process_files(max_files=None)  # Process 10 files for testing
    save_results(results)
    
    # Print summary
    valid_count = sum(1 for r in results.values() if r["valid"])
    print("\nSummary:")
    print(f"Total files processed: {len(results)}")
    print(f"Valid language files: {valid_count} ({(valid_count/len(results)*100 if results else 0):.1f}%)")
    print(f"Invalid/Unreadable files: {len(results) - valid_count}")

if __name__ == "__main__":
    main()