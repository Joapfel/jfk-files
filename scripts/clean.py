import argparse
import json
import logging
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# --- Configuration & Setup ---
DATA_DIR = Path("../data")
PARTS = ["Part 1", "Part 2", "Part 3"]

# --- Regular Expressions ---
FILENAME_RE = re.compile(r"JFK-Files-Part-(\d+)_page_(\d+)\.md")
HEADER_RE = re.compile(r"^#\s*JFK-Files-Part-\d+_page_\d+\.(png|md|jpg)\s*\n", re.IGNORECASE | re.MULTILINE)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE
)
CHECKED_CLASS_RE = re.compile(
    r"\[\s*[Xx]\s*\]\s*(TOP SECRET|SECRET|CONFIDENTIAL|RESTRICTED|UNCLASSIFIED)\b",
    re.IGNORECASE
)
# Example Cryptonym normalizer (e.g., matches LITAMIL7 or LITAMIL 7 and groups them to easily format as LITAMIL-7)
CRYPTO_RE = re.compile(r"\b([A-Z]{3,7})\s?-?(\d{1,3})\b")
CRYPTO_STOPWORDS = {"DATE", "FORM", "PAGE", "PART", "ROOM", "COPY", "FILE", "DOC", "USE"}

def setup_logging():
    """Configures thread-safe logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Normalization Functions ---
def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to standardize OCR artifacts."""
    # NFKC normalizes ligatures and specialized characters (e.g., ﬁ -> fi)
    return unicodedata.normalize("NFKC", text)

def remove_boilerplate(text: str) -> str:
    """Strips out recurring OCR headers and artifacts."""
    # Remove the markdown image header injected during the OCR phase
    text = HEADER_RE.sub("", text)
    return text

def normalize_cryptonyms(text: str) -> str:
    """Standardizes CIA cryptonyms to a hyphenated format (e.g., LITAMIL-7)."""
    return CRYPTO_RE.sub(r"\1-\2", text)

def clean_text(text: str) -> str:
    """Applies all text cleaning and formatting rules."""
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = normalize_cryptonyms(text)
    
    # Reflow text: keep paragraph breaks (double newlines), but merge line breaks within paragraphs
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for p in paragraphs:
        # Replace single newlines with spaces, collapse multiple spaces
        p = re.sub(r"(?<!\n)\n(?!\n)", " ", p)
        p = re.sub(r"[ \t]+", " ", p).strip()
        if p:
            cleaned_paragraphs.append(p)
            
    return "\n\n".join(cleaned_paragraphs)

# --- Metadata Extraction ---
def extract_metadata(text: str, filename: str) -> dict:
    """Extracts metadata from the filename and the document text."""
    metadata = {
        "part": None,
        "page_image": None,
        "classifications": [],
        "dates_mentioned": [],
        "cryptonyms_mentioned": []
    }
    
    # Filename parsing (Stable Identifier)
    m = FILENAME_RE.search(filename)
    if m:
        metadata["part"] = int(m.group(1))
        metadata["page_image"] = int(m.group(2))
        
    # Text-based extraction
    metadata["classifications"] = list(set(c.upper() for c in CHECKED_CLASS_RE.findall(text)))
    metadata["dates_mentioned"] = list(set(DATE_RE.findall(text)))

    # Extract standardized cryptonyms for the first-pass statistical collection
    cryptos = CRYPTO_RE.findall(text)
    if cryptos:
        metadata["cryptonyms_mentioned"] = list(set([
            f"{c[0]}-{c[1]}" for c in cryptos
            if c[0].upper() not in CRYPTO_STOPWORDS
        ]))
        
    return metadata

# --- Core Processing Logic ---
def process_file(file_path: Path, output_dir: Path, meta_dir: Path, force: bool = False):
    """Processes a single markdown file, normalizing it and extracting metadata."""
    try:
        normalized_file = output_dir / file_path.name
        metadata_file = meta_dir / f"{file_path.stem}.json"
        
        # Resumability check
        if not force and normalized_file.exists() and metadata_file.exists():
            return {"status": "skipped", "file": file_path.name}

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # 1. Extract metadata (using un-normalized and normalized text where appropriate)
        metadata = extract_metadata(text, file_path.name)
        
        # 2. Run normalization pipeline
        cleaned_text = clean_text(text)
        
        # 3. Write outputs
        normalized_file.write_text(cleaned_text, encoding="utf-8")
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        
        return {"status": "processed", "file": file_path.name}
    
    except Exception as e:
        return {"status": "error", "file": file_path.name, "error": str(e)}

# --- Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="JFK OCR Data Normalizer Pipeline")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing processed files")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU cores to use (default: 4)")
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting JFK Document Pipeline...")

    tasks = []
    
    # Discover all files to process
    for part in PARTS:
        part_dir = DATA_DIR / part
        input_dir = part_dir / "ocr-dots-md"
        output_dir = part_dir / "normalized-md"
        meta_dir = part_dir / "metadata"

        # Ensure directories exist
        if not input_dir.exists():
            logging.warning(f"Input directory missing: {input_dir}. Skipping.")
            continue
            
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob("JFK-Files-Part-*_page_*.md"))
        for f in files:
            tasks.append((f, output_dir, meta_dir))

    total_files = len(tasks)
    logging.info(f"Discovered {total_files} files across {len(PARTS)} parts.")
    
    # Process files concurrently
    processed_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, task[0], task[1], task[2], args.force): task
            for task in tasks
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            
            if result["status"] == "processed":
                processed_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
            else:
                error_count += 1
                logging.error(f"Error processing {result['file']}: {result['error']}")
                
            # Log progress periodically
            if i % 1000 == 0 or i == total_files:
                logging.info(f"Progress: {i}/{total_files} files evaluated...")

    logging.info("Pipeline Execution Complete.")
    logging.info(f"Total Processed: {processed_count}")
    logging.info(f"Total Skipped: {skipped_count}")
    logging.info(f"Total Errors: {error_count}")

if __name__ == "__main__":
    main()