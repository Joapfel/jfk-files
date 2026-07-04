# JFK Files — RAG Evaluation Testset

A set of 52 question-answer pairs for evaluating retrieval-augmented generation (RAG) over the declassified JFK files corpus. Generated with [RAGAS](https://docs.ragas.io/) using Claude as the LLM and a local sentence-transformers model for embeddings.

## Approach

1. **Corpus**: random sample of 150 pages from the OCR'd JFK files (`data/Part */ocr-dots-md/*.md`). Each file is one scanned page, already chunked at the page level.
2. **Knowledge graph**: RAGAS builds an in-memory knowledge graph from the sampled pages:
   - `SummaryExtractor` — one-paragraph summary per page (Claude)
   - `EmbeddingExtractor` — embeds the summaries with `all-MiniLM-L6-v2`
   - `ThemesExtractor` / `NERExtractor` — key themes and named entities per page (Claude)
   - `CosineSimilarityBuilder` — links pages with similar summaries (threshold 0.7)
   - `OverlapScoreBuilder` — links pages sharing named entities
3. **Synthesis**: three question types are generated from the graph, roughly equally weighted (~17 each):
   - `single_hop_specific` — answerable from a single page, anchored to specific entities
   - `multi_hop_abstract` — requires combining themes across related pages
   - `multi_hop_specific` — requires combining named entities across related pages

The resulting questions are grounded: each `reference` answer and each `reference_contexts` entry was derived directly from the source pages, so the testset can be used for both retrieval and answer-quality evaluation.

## Files

| File | Description |
|---|---|
| `testset.json` | Evaluation testset — 52 QA pairs (primary format) |
| `testset.csv` | Same data as CSV |
| `knowledge_graph.json` | RAGAS knowledge graph built from the 150-page sample; can be reused to regenerate the testset without re-running all LLM calls (`SKIP_KG_BUILD=1`) |
| `generate_testset.py` | Script that produced the testset (see below) |
| `requirements.txt` | Python dependencies |

## testset.json — field reference

Each entry is a JSON object:

```json
{
  "user_input": "...",
  "reference_contexts": ["...", "..."],
  "reference": "...",
  "synthesizer_name": "...",
  "persona_name": "...",
  "query_style": "...",
  "query_length": "..."
}
```

| Field | Type | Description |
|---|---|---|
| `user_input` | string | The question — use this as the query to your retriever / RAG pipeline |
| `reference_contexts` | list of strings | The source page(s) the answer was grounded in. Each string is the full text of one OCR'd page. Use these as ground-truth for retrieval metrics (`context_recall`, `context_precision`). |
| `reference` | string | The expected answer. Use this for answer-quality metrics (`answer_correctness`, `faithfulness`). |
| `synthesizer_name` | string | Which RAGAS synthesizer produced the pair (`single_hop_specific_query_synthesizer`, `multi_hop_abstract_query_synthesizer`, `multi_hop_specific_query_synthesizer`) |
| `persona_name` | string | Persona the question was written from (single-hop only; null for multi-hop) |
| `query_style` | string | Surface variation applied to the question, e.g. `MISSPELLED`, `WEB_SEARCH_LIKE` (single-hop only; null for multi-hop) |
| `query_length` | string | `SHORT`, `MEDIUM`, or `LONG` (single-hop only; null for multi-hop) |

### What "one entry in reference_contexts" is

Each string in `reference_contexts` is the OCR text of a single scanned page — i.e. the content of one `.md` file from `ocr-dots-md/`. For retrieval evaluation, your search system should return these pages (or passages that overlap with them) when given the corresponding `user_input`.

## Regenerating the testset

```bash
pip install -r testset/requirements.txt

# Full rebuild (150 docs × ~2 LLM calls each)
ANTHROPIC_API_KEY=<key> python testset/generate_testset.py

# Resume from a saved knowledge graph — skips summarisation, adds only
# missing transforms (NER, Themes, similarity), then runs synthesis
RESUME_KG_BUILD=1 ANTHROPIC_API_KEY=<key> python testset/generate_testset.py

# Skip KG build entirely — run synthesis on an already-complete KG
SKIP_KG_BUILD=1 ANTHROPIC_API_KEY=<key> python testset/generate_testset.py
```

Other env vars: `TESTSET_SIZE` (default 50), `DOCS_SAMPLE_SIZE` (default 150), `MIN_DOC_LENGTH` (default 400 chars), `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`).
