# jfk-files

OCR + cleaning + human-review pipeline for the NARA JFK document release
(~82k scanned pages). The stages below are a reusable template for turning a pile of
scanned PDFs into clean, embeddable text with reviewed quality.

## Pipeline

```
PDFs → PNGs → OCR (Markdown) → clean/normalize (txt + metadata) → dedup → language check → human review → feedback loop
```

Per-part data layout under `data/`:
`Part N/{ocr-dots-md, normalized, metadata}/JFK-Files-Part-N_page_M.{md,txt,json}`.
Source PDFs/PNGs live outside the repo at `<base>/JFK-Files-Part-N_pages/{pdf,png}/`.

| # | Stage | Script | In → Out |
|---|-------|--------|----------|
| 1 | **Acquire** | — | Download the release PDFs (one per page) to `<base>/JFK-Files-Part-N_pages/pdf/` |
| 2 | **PDF → PNG** | `pdf_to_png.py <pdf_dir> <png_dir> --dpi 300 --natural-sort` | per-page PDF → `{stem}.png` (300 DPI) |
| 3 | **OCR** | `ocr_dots.py` | PNG → Markdown in `Part N/ocr-dots-md/` (dots.ocr; alternatives tried: chandra/deepseek/nanonets/docling) |
| 4 | **Clean / normalize** | `clean.py [--force\|--soft-force] [--workers N]` | `ocr-dots-md/*.md` → `normalized/*.txt` + `metadata/*.json` |
| 5 | **Deduplicate** | `auto_confirm_duplicates.py`, `find_near_duplicates.py` | collapse identical / stamp-only-different pages |
| 6 | **Language check** | `language_validity_normalized.py` | `normalized/*.txt` → `data/validity/languageValidity_normalized.json` (langdetect) |
| 6b | **Classify empties (pixel)** | `rank_empty_pages.py` | render `[EMPTY]` source PDFs → `data/review/empty_ink.json` (ink density + first-pass category; reliably tags `blackened`) |
| 6c | **Embed empties (SigLIP)** | `embed_empty_pages.py` | `[EMPTY]` PDFs → `data/review/empty_embed.npz` (SigLIP-2 image embeddings, GPU) |
| 6d | **Re-categorise (nearest-centroid)** | `reclassify_empty_pages.py` | embeddings → cluster centroids (`cat_centroids.npz`) → better blank/text/photo split written back to `empty_ink.json` |
| 7 | **Human review** | `review_app.py` (local web UI at :8000) | side-by-side PDF/text/metadata → decisions in `data/review/reviews.json` |
| 8 | **Feedback loop** | `/review-feedback` skill | NOK feedback → improve `clean.py` → `clean.py --soft-force` → re-check |

## Stage detail

**4. clean.py** — the core normalizer. Strips administrative noise (stamps, routing
blocks, form furniture, classification markings, copy/file stamps, markdown/HTML) and
routes structured data to `metadata/*.json` (doc_id, classifications, dates, file numbers,
document_type/date, routing-sheet TO/COMMENTS, cover-sheet subjects, duplicate_of, …). It
also reshapes index-card pages: **cover sheets** (FBI/State "DATE / SUBJECT / FILE NO" cards,
incl. the "FBI doc. / RE:" dialect) route their catalog fields to metadata while keeping the
SUBJECT lines as the embeddable text; **name-index** pages drop their page-number runs but
keep the names. Non-content pages become a single **indicator token** (whole normalized file)
so an embedder can skip them:

| Indicator | Meaning | Set by |
|-----------|---------|--------|
| `[EMPTY]` | blank / stub page (only a box number, etc.) | `clean.py` |
| `[IDENTIFICATION FORM]` | NARA identification cover form | `clean.py` |
| `[ROUTING AND RECORD SHEET]` | pure routing-sheet overhead | `clean.py` |
| `[DISTRIBUTION LIST]` | pure component/station distribution or dissemination manifest | `clean.py` |
| `[COVER SHEET]` | source-document index card with no SUBJECT (all catalog fields → metadata) | `clean.py` |
| `[DUPLICATE OF <stem>]` | duplicate of a kept canonical page | dedup tools |
| `[UNREADABLE]` | source illegible | reviewer (Unreadable button) |
| `[FORM]` | a form with no real information | reviewer (Form button) |
| `[REOCR]` | source is readable but OCR failed — re-run OCR later | reviewer (Re-OCR button) |
| `[PHOTO]` | page is a photograph / image (no OCR-able text) | reviewer (No-content grid) |
| `[BLACKENED]` | page is predominantly redacted / blacked-out | reviewer (No-content grid) |

- `--force` reprocess everything · `--soft-force` reprocess everything **except** pages a
  human endorsed (`ok`/`unreadable`/`empty`/`form`/`reocr` in `reviews.json`) so review work
  and reviewer-set indicators are never lost.

**5. Dedup** — `auto_confirm_duplicates.py` collapses byte-identical (alnum) pages;
`find_near_duplicates.py` collapses pages that differ **only by a stamp/page number**
(token-Jaccard + stamp-vocabulary safety). Both keep one canonical, mark the rest
`[DUPLICATE OF …]`, write `duplicate_of` to every member's metadata, and log to
`reviews.json` (`--dry-run` first; `reviews.json.bak` backup).

**7. review_app.py** — human-in-the-loop UI. Shows the original PDF page next to the
normalized text, raw markdown, and metadata. Per-page decisions: **OK** / **NOK**+feedback /
**Unreadable** (`[UNREADABLE]`) / **Empty** (`[EMPTY]`) / **Form** (`[FORM]`) /
**Re-OCR** (`[REOCR]`) / **Foreign** (valid foreign-language, text kept). Session modes to
shrink the queue:
- **Sequential / Sample N** — general review.
- **Duplicates** — one decision per identical group (keeps a canonical, dedups the rest).
- **No-content** — batch-approve the `[EMPTY]/[…FORM]/[ROUTING]/[UNREADABLE]` pages.
- **No-content grid** — thumbnail grid of the `[EMPTY]` pages **auto-sorted into categories**
  (blank / text / photo / blackened). `blackened` comes from pixel stats; the blank/text/photo
  split comes from **SigLIP-2 image embeddings** (stages 6c–6d) — pixel stats and zero-shot
  text prompts could not separate degraded blank/text/photo scans, but the image-embedding
  space clusters them by visual type (photos, weapons, maps vs. typed vs. handwritten vs.
  blank). A dropdown shows one category at a time (densest-first) so a visually homogeneous
  bucket can be judged at a glance; every page is pre-labeled with its indicator
  (`[EMPTY]`/`[REOCR]`/`[PHOTO]`/`[BLACKENED]`), click a thumbnail to reassign an outlier,
  then **Apply** marks the whole bucket. The already-typed structural pages
  (`[IDENTIFICATION FORM]`, `[ROUTING AND RECORD SHEET]`, `[UNREADABLE]`) appear as their own
  categories too and default to **KEEP** (approve, leaving the indicator in place). Turns
  ~3k one-at-a-time decisions into a few sweeps.
- **Low-validity** — review only pages the language check flagged with confidence < 0.5
  (likely gibberish; confident foreign-language pages are excluded as valid).
- **Garbage** — pages the pipeline flagged `suspected_garbage` (OCR-hallucinated math).
- **Re-check NOK** — re-verify previously-flagged pages after a fix (`resolved` flag).

## Design principles (reusable)
- **Two outputs per page**: clean embeddable text + structured metadata JSON.
- **Indicators, not deletion** — content-free pages get a `[TAG]` marker; nothing is destroyed
  (raw `.md` + PDFs are the source of truth; every automated step is reversible).
- **Automate the certain, surface the uncertain** — dedup, structural, and validity gates clear
  the obvious cases; humans review only the ambiguous remainder.
- **Feedback loop** — reviewer NOK notes drive concrete `clean.py` fixes, re-applied with
  `--soft-force` so approved pages stay put.

## Quick start
```bash
conda activate jfk-files
cd scripts
python pdf_to_png.py "<base>/JFK-Files-Part-3_pages/pdf/" "<base>/JFK-Files-Part-3_pages/png/" --natural-sort
python ocr_dots.py                 # PNG → ocr-dots-md
python clean.py --workers 8        # → normalized/ + metadata/
python language_validity_normalized.py
python review_app.py               # open http://127.0.0.1:8000  (set the PDF base path in the UI)
```
