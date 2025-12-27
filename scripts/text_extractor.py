import os
from pypdf import PdfReader

# Base directory containing the JFK files
base_path = r"C:\Users\<user>\Documents\John F Kennedy Files"
parts = ["JFK-Files-Part-1_pages", "JFK-Files-Part-2_pages", "JFK-Files-Part-3_pages"]

def ensure_txt_directory(part_path):
    """Create a txt directory in the part directory if it doesn't exist."""
    txt_dir = os.path.join(part_path, 'txt')
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
        print(f"Created directory: {txt_dir}")
    return txt_dir

def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def main():
    for part in parts:
        part_path = os.path.join(base_path, part)
        if not os.path.exists(part_path):
            print(f"Directory not found: {part_path}")
            continue
            
        print(f"\nProcessing files in: {part}")
        txt_dir = ensure_txt_directory(part_path)
        
        # Iterate through all files in the part directory
        for filename in os.listdir(part_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(part_path, filename)
                print(f"Extracting text from: {filename}")
                
                # Process the PDF and get the text
                text = process_pdf(file_path)
                
                # Save to individual text file in the txt directory
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_filepath = os.path.join(txt_dir, txt_filename)
                
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"  - Saved text to: {txt_filepath}")
    
    print("\nExtraction complete. Text files have been saved in the 'txt' directories.")

if __name__ == "__main__":
    main()