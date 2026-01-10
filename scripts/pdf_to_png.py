#!/usr/bin/env python3
"""
PDF to PNG Converter for Batch Processing
Converts all PDF files in a directory to PNG images, one page per PNG.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import concurrent.futures
from tqdm import tqdm

try:
    from pdf2image import convert_from_path
    from PIL import Image
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install with: pip install pdf2image pillow PyMuPDF tqdm")
    sys.exit(1)


class PDFToPNGConverter:
    def __init__(self, dpi: int = 300, output_format: str = "PNG", quality: int = 95):
        """
        Initialize the PDF to PNG converter.
        
        Args:
            dpi: Resolution for conversion (higher = better quality, larger files)
            output_format: Output format (PNG, JPEG)
            quality: Quality for JPEG (1-100)
        """
        self.dpi = dpi
        self.output_format = output_format.upper()
        self.quality = quality
        
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str, prefix: str = None) -> List[str]:
        """
        Convert a single PDF file to PNG images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save the PNG files
            prefix: Optional prefix for output filenames
            
        Returns:
            List of paths to the created image files
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate prefix from PDF filename if not provided
        if prefix is None:
            prefix = pdf_path.stem
        
        # Remove spaces and special characters from prefix
        prefix = "".join(c for c in prefix if c.isalnum() or c in ('-', '_'))
        
        try:
            # Try using pdf2image first (better quality)
            images = convert_from_path(pdf_path, dpi=self.dpi, output_folder=output_dir, 
                                     output_file=prefix, fmt=self.output_format.lower(),
                                     thread_count=4, use_pdftocairo=True)
            
            # Rename files to include page numbers
            output_files = []
            for i, image in enumerate(images):
                old_path = image.filename
                new_filename = f"{prefix}_page_{i+1}.{self.output_format.lower()}"
                new_path = output_dir / new_filename
                
                # pdf2image already saves the file, just rename it
                if old_path != new_path:
                    os.rename(old_path, new_path)
                
                output_files.append(str(new_path))
                
            return output_files
            
        except Exception as e:
            print(f"pdf2image failed for {pdf_path}: {e}")
            # Fallback to PyMuPDF
            return self._convert_with_pymupdf(pdf_path, output_dir, prefix)
    
    def _convert_with_pymupdf(self, pdf_path: str, output_dir: str, prefix: str) -> List[str]:
        """
        Convert PDF to images using PyMuPDF as fallback.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save the PNG files
            prefix: Prefix for output filenames
            
        Returns:
            List of paths to the created image files
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_files = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Convert each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get page dimensions
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                # Save as PNG
                output_filename = f"{prefix}_page_{page_num + 1}.{self.output_format.lower()}"
                output_path = output_dir / output_filename
                
                if self.output_format == "PNG":
                    pix.save(output_path)
                else:  # JPEG
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save(output_path, quality=self.quality)
                
                output_files.append(str(output_path))
            
            doc.close()
            return output_files
            
        except Exception as e:
            print(f"PyMuPDF also failed for {pdf_path}: {e}")
            return []
    
    def convert_directory(self, input_dir: str, output_dir: str, 
                        max_workers: int = 4, recursive: bool = False) -> dict:
        """
        Convert all PDF files in a directory to PNG images.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save PNG files
            max_workers: Number of parallel workers
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with conversion statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find all PDF files
        if recursive:
            pdf_files = list(input_dir.rglob("*.pdf"))
        else:
            pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return {"total_files": 0, "successful": 0, "failed": 0, "total_pages": 0}
        
        print(f"Found {len(pdf_files)} PDF files to convert")
        
        stats = {"total_files": len(pdf_files), "successful": 0, "failed": 0, "total_pages": 0}
        
        # Convert files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pdf = {}
            for pdf_file in pdf_files:
                # Create subdirectory for each PDF if recursive
                if recursive:
                    relative_path = pdf_file.relative_to(input_dir)
                    pdf_output_dir = output_dir / relative_path.parent
                else:
                    pdf_output_dir = output_dir
                
                future = executor.submit(self.convert_pdf_to_images, pdf_file, pdf_output_dir)
                future_to_pdf[future] = pdf_file
            
            # Process results with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_pdf), 
                             total=len(pdf_files), desc="Converting PDFs"):
                pdf_file = future_to_pdf[future]
                try:
                    output_files = future.result()
                    if output_files:
                        stats["successful"] += 1
                        stats["total_pages"] += len(output_files)
                        print(f"✓ Converted {pdf_file.name} -> {len(output_files)} pages")
                    else:
                        stats["failed"] += 1
                        print(f"✗ Failed to convert {pdf_file.name}")
                except Exception as e:
                    stats["failed"] += 1
                    print(f"✗ Error converting {pdf_file.name}: {e}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to PNG images")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("output_dir", help="Directory to save PNG files")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for conversion (default: 300)")
    parser.add_argument("--format", choices=["PNG", "JPEG"], default="PNG", help="Output format (default: PNG)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--prefix", help="Prefix for output filenames (default: PDF filename)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Create converter
    converter = PDFToPNGConverter(dpi=args.dpi, output_format=args.format, quality=args.quality)
    
    # Convert files
    print(f"Converting PDF files from '{args.input_dir}' to '{args.output_dir}'")
    print(f"Settings: DPI={args.dpi}, Format={args.format}, Workers={args.workers}")
    
    stats = converter.convert_directory(
        args.input_dir, 
        args.output_dir, 
        max_workers=args.workers,
        recursive=args.recursive
    )
    
    # Print summary
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Total PDF files: {stats['total_files']}")
    print(f"Successfully converted: {stats['successful']}")
    print(f"Failed conversions: {stats['failed']}")
    print(f"Total pages created: {stats['total_pages']}")
    
    if stats['failed'] > 0:
        print(f"\nWarning: {stats['failed']} files failed to convert")
        sys.exit(1)
    else:
        print("\nAll files converted successfully!")


if __name__ == "__main__":
    main()