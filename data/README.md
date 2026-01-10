The data directory contains the JFK files in text format. Due to their size the original PDF files are not included into this repository, you can find them [here](https://www.whitehouse.gov/jfk-files/).

To retrieve the OCR-based text provided in the original files, the large PDF files were split into smaller files (one file per page) using [pdfcpu](https://pdfcpu.io/extract/extract_pages.html). The text was retrieved by the text_extractor.py script. The "Part" structure and mapping to PDF pages is preserved for traceability.

Language Detection References:
- langdetec, see [Presentation](https://de.slideshare.net/slideshow/language-detection-library-for-java/6014274#27)
- FastText, see facebook's [Github](https://github.com/facebookresearch/fastText?tab=readme-ov-file)

As the original PDFs contain poor quality OCR, the [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) model was used to retrieve higher quality text. The `ocr_dots.py` script (which is an extension of [this Demo App](https://huggingface.co/blog/prithivMLmods/multimodal-ocr-vlms#ii-dotsocr)) was used to retrieve the text. As the script operates on images, the `pdf_to_png.py` script was used to convert the PDFs to images. The images are not included into this repository due to their size. The final text can be found in the `ocr-dots-md` directory.