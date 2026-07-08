# MCP performance instrumentation (`timing_ms`)

Every MCP tool response can include a top-level `timing_ms` object with millisecond timings for request handling and tool-specific stages. Timings use monotonic clocks and are enabled by default.

Disable output in production with:

```toml
[instrumentation]
timing_enabled = false
```

## Top-level fields (all tools)

| Field | Meaning |
|-------|---------|
| `total` | Wall time for the tool handler |
| `request_parse` | Parameter validation before main work |
| `routing` | Always `0` (rmcp routes before the handler runs) |
| `tool_execution` | Remaining handler work excluding serialization |
| `serialization` | JSON encoding of the response body |

## `search` tool — additional fields

| Field | Stage |
|-------|-------|
| `embedding_generation` | Query embedding via fastembed |
| `sqlite_fts` | FTS5 candidate retrieval |
| `vector_search` | sqlite-vec KNN candidate retrieval |
| `reciprocal_rank_fusion` | RRF merge of FTS + vector lists |
| `reranker` | Cross-encoder rerank (0 when off) |
| `mmr` | MMR diversification (0 when off) |
| `parent_retriever` | Display-time content expansion |
| `result_filtering` | `low_confidence` flag + `match_spans` |
| `response_build` | Assembling `SearchResponse` / filter echo |

Example excerpt:

```json
{
  "results": [ ... ],
  "low_confidence": false,
  "filter_applied": {},
  "timing_ms": {
    "total": 842,
    "request_parse": 0,
    "routing": 0,
    "tool_execution": 820,
    "serialization": 22,
    "embedding_generation": 45,
    "sqlite_fts": 12,
    "vector_search": 18,
    "reciprocal_rank_fusion": 1,
    "reranker": 0,
    "mmr": 0,
    "parent_retriever": 0,
    "result_filtering": 2,
    "response_build": 0
  }
}
```

If `embedding_generation` dominates, the embedder or model choice is the bottleneck. If `sqlite_fts` / `vector_search` dominate, inspect index size and filters. A high `get_document` `disk_read` with `cache_lookup: 0` points at cold disk I/O (document missing from the in-memory index).

## `get_document` tool — additional fields

`get_document(path)` is **deterministic direct retrieval**: it looks up the canonical relative path in an in-memory document index built at server startup (and kept in sync by `rebuild_index` / the file watcher). It does not run the search pipeline, generate embeddings, or touch sqlite-vec.

| Field | Stage |
|-------|-------|
| `document_lookup` | Path key validation (no disk I/O) |
| `cache_lookup` | In-memory index probe |
| `disk_read` | Always `0` on the hot path (served from memory) |
| `frontmatter_parse` | Always `0` on the hot path (parsed at index time) |
| `markdown_load` | Legacy field; always `0` when served from cache |
| `response_build` | Assembling the JSON response from the cached entry |

Latency regression test: `cargo test --test document_index_integration -- --ignored` (reports median/p95 over 20 repeated fetches).

## `list_topics` tool

Response shape is `{ "topics": [ ... ], "timing_ms": { ... } }` with:

| Field | Stage |
|-------|-------|
| `topic_index_lookup` | `db.list_topics()` |
| `response_build` | Mapping rows to JSON entries |

## Interpreting totals

Stage timings are sequential buckets and may overlap slightly with `tool_execution`; they are meant for relative comparison, not exact accounting. `total` should equal the sum of the five top-level fields.
