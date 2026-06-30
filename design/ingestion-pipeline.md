# Ingestion Pipeline — Design

_Last updated: 2026-06-29_

---

## About This File

Source of truth for high-level design decisions and inter-stage contracts.

### Sections and When to Update Them

| Section | Update when |
|---|---|
| **Pipeline Overview** | The pipeline shape changes (new stage, new mode, orchestrator wiring changes) |
| **Package Structure** | Modules or files are added, moved, or renamed |
| **Component Responsibilities** | Ownership boundaries between components change |
| **Stage Interfaces** | A stage's public entry points or output schema change — this is the inter-stage contract that all other stages depend on |
| **Deferred** | A design decision is postponed (add here) or is ready to implement |

---

## Pipeline Overview

The ingestion pipeline moves content from external sources into Weaviate
in four stages:

```
Source (API / files)
        │
        ▼
   [ Scraper ]  ──→  {source_key}.jsonl         (optional, file mode only)
        │
        ▼
   [ Chunker ]  ──→  {source_key}_chunked.jsonl  (optional, file mode only)
        │
        ▼
   [ Ingestor ] ──→  Weaviate
```

Each stage has a single well-defined IO contract — plain dicts in canonical
format. No stage knows about the internals of any other.

### Modes

| Mode | Description |
|---|---|
| **File mode** | Scraper writes `{source_key}.jsonl`; Chunker reads it and writes `{source_key}_chunked.jsonl`; Ingestor reads that. Intermediate files allow re-chunking or re-ingesting without re-scraping. |
| **Streaming mode** | Scraper yields canonical dicts directly to the Chunker with no intermediate file. The Chunker still writes a chunked JSONL for the Ingestor. Faster end-to-end; no raw JSONL on disk. |

The **Orchestrator** owns the mode decision and wires the stages together.
Only `{source_key}_items.json` is always written — it is the resume and
progress tracking file, independent of mode.

```python
if mode == "stream":
    # no intermediate file written; data flows in memory
    chunker.chunk_file(source=scraper.stream(), out=chunked_jsonl)
else:
    # file mode: write raw JSONL first, then chunk from it
    if not raw_jsonl.exists():
        scraper.scrape()  # writes {source_key}.jsonl
    chunker.chunk_file(source=raw_jsonl, out=chunked_jsonl)
```

The file existence check only applies in file mode — in streaming mode there is
no intermediate file to reuse. Further orchestrator design is deferred.

---

## Package Structure

```
rubin_rag/
  ingestion_pipeline/
    scrapers/        # fetch raw content per source, write raw JSONL
      base.py        # BaseScraper ABC + ItemsManifest
      scrape_confluence.py
    chunking/        # source-agnostic chunking
      base.py        # BaseChunkingStrategy ABC + STRATEGY_REGISTRY
      chunker.py     # Chunker class
      strategies.py  # all concrete strategy classes
    orchestrator.py  # wires stages together, owns all file I/O
    ingestor.py      # reads chunked JSONL, batches by tokens, pushes to Weaviate
  chatbot/           # query Weaviate, LLM interaction, Streamlit app
```

---

## Component Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| **Scraper** | Fetching content, writing raw JSONL, yielding canonical dicts | Mode decision, chunking, ingestion |
| **Chunker** | Splitting canonical dicts into chunks, writing chunked JSONL | Source awareness, strategy selection, mode decision |
| **Strategy** | Text splitting algorithm and its parameters | Record format, metadata, chunk assembly |
| **Orchestrator** | Mode decision, output paths, wiring components together | Write mechanics, source-specific logic, splitting logic |
| **Ingestor** | Token batching, embedding API calls, Weaviate writes | Scraping, chunking |

---

## Stage Interfaces

All stages exchange data as plain dicts — no library-specific types cross stage
boundaries. This section defines the public entry points each stage exposes and
the output schema it produces. Internal implementation details are in
[`implementation_overview/`](implementation_overview/ingestion-pipeline-architecture.md).

### Scraper

Entry points called by the orchestrator:

| Method | Mode | Description |
|---|---|---|
| `scrape(max_pages=None)` | file | Writes `{source_key}.jsonl`. Self-contained; supports resume via manifest. |
| `stream()` | streaming | Lazy generator — yields canonical dicts directly to the caller. No file written. |

Output — one dict per document:

```json
{
  "text": "...",
  "metadata": {
    "source": "https://...",
    "source_key": "confluence",
    "item_key": "SP/152503124",
    "source_metadata": { ... }
  }
}
```

`source_metadata` keys vary by source (e.g. Confluence adds `space_key`,
`wiki_url`, `page_id`, `page_title`, and optionally `when_edited`). See the
source's implementation file for its specific shape.

### Chunker

Entry point called by the orchestrator:

| Method | Input | Description |
|---|---|---|
| `chunk_file(source, out_path)` | `Path` or `Iterator[dict]` | Reads source (JSONL file or live stream from `scraper.stream()`), writes `{source_key}_chunked.jsonl`. |

Output — all scraper fields preserved; two fields added to `metadata`:

```json
{
  "text": "<chunk text>",
  "metadata": {
    "source": "https://...",
    "source_key": "confluence",
    "item_key": "SP/152503124",
    "chunk_index": 0,
    "chunking_strategy": "recursive_character",
    "source_metadata": { ... }
  }
}
```

### Output Files

```
output_dir/
  {source_key}_items.json      # always written — manifest and progress state
  {source_key}.jsonl           # file mode only — raw scraper output
  {source_key}_chunked.jsonl   # chunker output
```

---

## Deferred

- **Orchestrator** — not yet implemented. Mode decision logic (stream vs. file,
  skip-rescrape condition) to be finalised during orchestrator design.
- **Ingestor** — `batch_by_tokens` and Weaviate push not yet restructured.
- **Additional scrapers** — GitHub, Jira, Discourse, etc. not yet ported to new ABC.
- **Pydantic config validation** — components currently accept raw `dict` and
  trust the config is correct. Migration path: define pydantic models per source
  → validate at load time in the orchestrator → swap `dict` parameter types to
  typed models (no internal logic changes).
- **Incremental scraping** — detect new/changed content since last run.
- **LangChain migration** — replace `ConfluenceLoader` with direct API calls.
