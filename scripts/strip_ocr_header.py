#!/usr/bin/env python3
"""Strip the deterministic dots.ocr furniture from the raw OCR layer (data/*/ocr-dots-md/*.md).

Every page dots.ocr produced is prefixed with two artifacts that are not document content and
that leak into the iPhone app's keyword-search previews:

    # JFK-Files-Part-1_page_1.png       <- H1 header echoing the source image filename
                                         (blank line)
    Convert to Markdown104-10003-10041  <- the literal OCR prompt, glued to the first token

This removes exactly those two (nothing else), in place. The files are git-tracked, so the
change is fully reversible (`git checkout -- data/*/ocr-dots-md`). Idempotent: re-running is a
no-op. The normalized/ pipeline already strips these via clean.py's HEADER_RE; this brings the
raw layer the app actually indexes in line.

Usage:
    python strip_ocr_header.py            # strip across all three parts
    python strip_ocr_header.py --dry-run  # report what would change, write nothing
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# same header pattern clean.py uses, anchored to the top of the file
HEADER_RE = re.compile(r"^#\s*JFK-Files-Part-\d+_page_\d+\.(png|md|jpg)\s*\n",
                       re.IGNORECASE | re.MULTILINE)
PROMPT = "Convert to Markdown"


def strip(text):
    """Remove every filename-H1 echo and the leading 'Convert to Markdown' prompt; leave content.
    The `# JFK-Files-Part-N_page_M.png` pattern only ever matches the OCR filename echo (never real
    1960s document text), so removing all occurrences — not just the first — is safe and keeps this
    idempotent for pages where the echo appears more than once."""
    new = HEADER_RE.sub("", text).lstrip()
    if new.startswith(PROMPT):
        new = new[len(PROMPT):].lstrip(" \t")     # drop the glued prompt, keep the content token
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    changed = scanned = 0
    for part in (1, 2, 3):
        d = DATA / f"Part {part}" / "ocr-dots-md"
        if not d.is_dir():
            continue
        for f in d.glob("*.md"):
            scanned += 1
            text = f.read_text(encoding="utf-8", errors="ignore")
            new = strip(text)
            if new != text:
                changed += 1
                if not args.dry_run:
                    f.write_text(new, encoding="utf-8")
    verb = "would change" if args.dry_run else "changed"
    print(f"scanned {scanned} files, {verb} {changed} ({scanned - changed} already clean / no artifact)")


if __name__ == "__main__":
    main()
