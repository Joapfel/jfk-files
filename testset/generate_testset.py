"""
Generate a RAGAS test set from JFK files OCR corpus.

Usage:
    pip install -r testset/requirements.txt
    ANTHROPIC_API_KEY=<key> python testset/generate_testset.py

Outputs (all written into the testset/ directory):
    testset/testset.csv            — user_input, reference, contexts, synthesizer_name
    testset/testset.json           — same data as JSON array
    testset/knowledge_graph.json   — reusable KG (skip rebuild on reruns)

Config env vars:
    ANTHROPIC_API_KEY   — required
    TESTSET_SIZE        — number of QA pairs to generate (default: 50)
    DOCS_SAMPLE_SIZE    — documents to sample from corpus (default: 150)
    MIN_DOC_LENGTH      — min chars to keep a document (default: 400)
    EMBEDDING_MODEL     — sentence-transformers model (default: all-MiniLM-L6-v2)
    SKIP_KG_BUILD       — set to 1 to load existing knowledge_graph.json and skip rebuild
"""

import os
import random
import pathlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Config ────────────────────────────────────────────────────────────────────
TESTSET_SIZE     = int(os.getenv("TESTSET_SIZE", "50"))
DOCS_SAMPLE_SIZE = int(os.getenv("DOCS_SAMPLE_SIZE", "150"))
MIN_DOC_LENGTH   = int(os.getenv("MIN_DOC_LENGTH", "400"))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SKIP_KG_BUILD    = os.getenv("SKIP_KG_BUILD", "0") == "1"
RESUME_KG_BUILD  = os.getenv("RESUME_KG_BUILD", "0") == "1"  # load KG, add missing transforms only

BASE_DIR    = pathlib.Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
OUT_DIR     = pathlib.Path(__file__).parent   # testset/ itself
KG_PATH     = OUT_DIR / "knowledge_graph.json"
CORPUS_DIRS = [
    DATA_DIR / "Part 1" / "ocr-dots-md",
    DATA_DIR / "Part 2" / "ocr-dots-md",
    DATA_DIR / "Part 3" / "ocr-dots-md",
]

# ── Imports ───────────────────────────────────────────────────────────────────
print("Loading libraries…")
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, NodeType
from ragas.testset.transforms import (
    EmbeddingExtractor,
    SummaryExtractor,
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
    Parallel,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor

# ── Patch ragas JSON output parser ────────────────────────────────────────────
# ragas 0.4.3 expects Claude to return JSON for every PydanticPrompt output
# (including plain StringIO→StringIO summaries). Claude often returns prose
# instead, crashing the transform phase (which has no raise_exceptions guard).
#
# Fix: when all JSON-parsing attempts fail, fall back by output model type:
#   StringIO             → wrap raw text as StringIO(text=...)
#   ThemesAndConcepts    → extract a list of strings from the prose
#   NEROutput            → same
# All other output models still raise so real bugs surface.
import json as _json
import re as _re
from ragas.prompt.pydantic_prompt import RagasOutputParser
from ragas.prompt import StringIO as RagasStringIO
from ragas.testset.transforms.extractors.llm_based import ThemesAndConcepts, NEROutput


def _extract_list_from_prose(text: str) -> list:
    """Best-effort extraction of a list of items from LLM prose output."""
    # 1. Try direct JSON parse
    try:
        obj = _json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj if x]
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return [str(x) for x in v if x]
    except (_json.JSONDecodeError, ValueError):
        pass
    # 2. Try to find a JSON array anywhere in the text
    m = _re.search(r'\[([^\[\]]{1,500})\]', text)
    if m:
        try:
            arr = _json.loads(f'[{m.group(1)}]')
            return [str(x) for x in arr if x]
        except (_json.JSONDecodeError, ValueError):
            pass
    # 3. Extract quoted strings (often entity names in prose)
    quoted = _re.findall(r'"([^"]{2,60})"', text)
    if quoted:
        return quoted[:15]
    # 4. Capitalised multi-word phrases (proper nouns / named entities)
    caps = _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    unique_caps = list(dict.fromkeys(caps))
    if unique_caps:
        return unique_caps[:15]
    return []


_orig_parse_output_string = RagasOutputParser.parse_output_string


async def _lenient_parse_output_string(self, output_string, prompt_value, llm, callbacks, retries_left=1):
    try:
        return await _orig_parse_output_string(self, output_string, prompt_value, llm, callbacks, retries_left)
    except Exception:
        if self.pydantic_object is RagasStringIO:
            return RagasStringIO(text=output_string)
        if self.pydantic_object is ThemesAndConcepts:
            return ThemesAndConcepts(output=_extract_list_from_prose(output_string))
        if self.pydantic_object is NEROutput:
            return NEROutput(entities=_extract_list_from_prose(output_string))
        raise


RagasOutputParser.parse_output_string = _lenient_parse_output_string
# ─────────────────────────────────────────────────────────────────────────────


def _build_transforms(llm, embedding_model):
    """
    Transform pipeline for the JFK pre-chunked corpus (one page per document).

    Omits HeadlineSplitter (documents are already single-page chunks).
    Omits CustomNodeFilter (uses a Pydantic output model that Claude also
    refuses to return as JSON; scoring each node isn't worth the extra calls).
    NER and Themes are included because all three default synthesizers require
    either entities or themes to find candidate node clusters.
    """
    def filter_docs(node):
        return node.type == NodeType.DOCUMENT

    return [
        SummaryExtractor(llm=llm, filter_nodes=filter_docs),
        Parallel(
            EmbeddingExtractor(
                embedding_model=embedding_model,
                property_name="summary_embedding",
                embed_property_name="summary",
                filter_nodes=filter_docs,
            ),
            ThemesExtractor(llm=llm, filter_nodes=filter_docs),
            NERExtractor(llm=llm, filter_nodes=filter_docs),
        ),
        Parallel(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.7,
                filter_nodes=filter_docs,
            ),
            OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_docs),
        ),
    ]


# ── Document loading ──────────────────────────────────────────────────────────
def load_docs(sample_size: int, min_length: int) -> list[Document]:
    all_files: list[pathlib.Path] = []
    for d in CORPUS_DIRS:
        if d.exists():
            all_files.extend(d.glob("*.md"))
        else:
            print(f"  Warning: {d} not found, skipping")

    if not all_files:
        raise RuntimeError("No .md files found — check that ocr-dots-md directories exist")

    random.seed(42)
    random.shuffle(all_files)

    docs: list[Document] = []
    skipped = 0
    for path in all_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()

        # Strip the image-name header ("# JFK-Files-Part-X_page_Y.png")
        lines = text.splitlines()
        if lines and lines[0].startswith("# JFK-Files"):
            text = "\n".join(lines[1:]).strip()

        # Strip "Convert to Markdown" artifact from dots OCR
        if text.startswith("Convert to Markdown"):
            text = text[len("Convert to Markdown"):].strip()

        if len(text) < min_length:
            skipped += 1
            continue

        part = path.parts[-3]  # "Part 1" / "Part 2" / "Part 3"
        docs.append(Document(
            page_content=text,
            metadata={"source": str(path), "part": part, "filename": path.name},
        ))
        if len(docs) >= sample_size:
            break

    print(f"Loaded {len(docs)} documents ({skipped} skipped, min_length={min_length})")
    return docs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    # LLM — Claude Sonnet (cost-effective, high quality)
    print("Initialising Claude LLM…")
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=4096,
    )

    # Embeddings — local sentence-transformers (no extra API key needed)
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads on first run)…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Context hint so the LLM generates domain-relevant questions
    llm_context = (
        "These documents are declassified US government intelligence files related to "
        "the assassination of President John F. Kennedy, released under the JFK Records "
        "Act. They include CIA and FBI reports, witness statements, surveillance records, "
        "and investigative documents from the 1960s–1990s. Generate realistic questions "
        "that a researcher or investigator might ask when searching through these records."
    )

    # Build generator
    print("Creating TestsetGenerator…")
    if RESUME_KG_BUILD and KG_PATH.exists():
        # KG already has summaries + embeddings; apply only the missing transforms
        # (NER, Themes, relationship builders) so we skip re-summarising 150 docs.
        from ragas.testset.transforms import apply_transforms
        from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
        from ragas.testset.transforms import OverlapScoreBuilder, CosineSimilarityBuilder, Parallel, EmbeddingExtractor
        print(f"Loading existing knowledge graph from {KG_PATH}…")
        kg = KnowledgeGraph.load(str(KG_PATH))
        generator = TestsetGenerator.from_langchain(
            llm, embeddings, knowledge_graph=kg, llm_context=llm_context
        )
        def filter_docs(node):
            return node.type == NodeType.DOCUMENT
        missing_transforms = [
            Parallel(
                ThemesExtractor(llm=generator.llm, filter_nodes=filter_docs),
                NERExtractor(llm=generator.llm, filter_nodes=filter_docs),
            ),
            Parallel(
                CosineSimilarityBuilder(
                    property_name="summary_embedding",
                    new_property_name="summary_similarity",
                    threshold=0.7,
                    filter_nodes=filter_docs,
                ),
                OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_docs),
            ),
        ]
        print("Applying missing transforms (NER, Themes, similarity)…")
        apply_transforms(kg, missing_transforms)
        generator.knowledge_graph.save(str(KG_PATH))
        print(f"Updated knowledge graph saved → {KG_PATH}")
        docs = None
    elif SKIP_KG_BUILD and KG_PATH.exists():
        print(f"Loading existing knowledge graph from {KG_PATH}…")
        kg = KnowledgeGraph.load(str(KG_PATH))
        generator = TestsetGenerator.from_langchain(
            llm, embeddings, knowledge_graph=kg, llm_context=llm_context
        )
        docs = None  # not needed when KG is prebuilt
    else:
        docs = load_docs(DOCS_SAMPLE_SIZE, MIN_DOC_LENGTH)
        generator = TestsetGenerator.from_langchain(
            llm, embeddings, llm_context=llm_context
        )

    # Generate — builds KG internally when docs are provided
    if docs is not None:
        print(
            f"Building knowledge graph + generating {TESTSET_SIZE} test pairs…\n"
            f"  (This calls Claude for each document — budget ~{DOCS_SAMPLE_SIZE * 2} API calls)\n"
        )
        transforms = _build_transforms(generator.llm, generator.embedding_model)
        testset = generator.generate_with_langchain_docs(
            documents=docs,
            testset_size=TESTSET_SIZE,
            transforms=transforms,
            raise_exceptions=False,
        )
        # Save KG for reuse (SKIP_KG_BUILD=1 on next run)
        generator.knowledge_graph.save(str(KG_PATH))
        print(f"Knowledge graph saved → {KG_PATH}")
    else:
        from ragas.testset.synthesizers import default_query_distribution
        print(f"Generating {TESTSET_SIZE} test pairs from cached KG…")
        testset = generator.generate(
            testset_size=TESTSET_SIZE,
            query_distribution=default_query_distribution(generator.llm),
            raise_exceptions=False,
        )

    # Convert to DataFrame and save
    eval_ds = testset.to_evaluation_dataset()
    df = eval_ds.to_pandas()

    # Add synthesizer names back from testset samples
    if "synthesizer_name" not in df.columns and testset.samples:
        df["synthesizer_name"] = [s.synthesizer_name for s in testset.samples[:len(df)]]

    csv_path  = OUT_DIR / "testset.csv"
    json_path = OUT_DIR / "testset.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"\nDone! {len(df)} test pairs written to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"\nColumns: {list(df.columns)}")
    if len(df) > 0:
        print(f"\nSample (first 3 rows):")
        cols = [c for c in ["user_input", "reference", "synthesizer_name"] if c in df.columns]
        print(df[cols].head(3).to_string())


if __name__ == "__main__":
    main()
