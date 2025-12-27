The data directory contains the JFK files in text format. Due to their size the original PDF files are not included into this repository, you can find them [here](https://www.whitehouse.gov/jfk-files/).

To retrieve the OCR-based text provided in the original files, the large PDF files were split into smaller files (one file per page) using [pdfcpu](https://pdfcpu.io/extract/extract_pages.html). The text was retrieved by the text_extractor.py script. The "Part" structure and mapping to PDF pages is preserved for traceability.

Language Detection References:
- langdetec, see [Presentation](https://de.slideshare.net/slideshow/language-detection-library-for-java/6014274#27)
- FastText, see facebook's [Github](https://github.com/facebookresearch/fastText?tab=readme-ov-file)