## Diagnosis

- Root cause: `get_document` in `kb-mcp/src/server.rs` performed per-request disk I/O (`validate_get_document_path` → `read_to_string` → parse) with no in-memory document cache, so known-path retrieval paid full filesystem + parse cost (~700 ms observed) even though agents only need a path lookup after search.
- Confirmed by: `kb-mcp/src/server.rs` (pre-change `get_document` handler used `std::fs::read_to_string`); `docs/perf-instrumentation.md` noted `cache_lookup: 0` until a cache exists; issue #2 acceptance criteria.
- Observability traps: `timing_ms` from #1 could show low `tool_execution` while disk syscalls dominated; a warm OS page cache could mask the problem locally.

## Proposed Fix

- Add `document_index` module: build an in-memory `HashMap` of canonical relative paths → parsed metadata + raw content at `run_server` startup; share via `Arc<RwLock<_>>` on `KbServerShared`.
- Rewrite `get_document` to validate the path key without disk I/O, lookup the index, and build the existing `DocumentResponse` shape; record `cache_lookup` / `response_build` timings.
- Refresh the index on `rebuild_index` completion and watcher create/modify/delete/rename events.
- Why not DB-backed lookup: SQLite still adds query latency and couples retrieval to search-index state; issue requires avoiding the search pipeline entirely and keeping hot path in memory.

## Files & Line Numbers

- `kb-mcp/src/document_index.rs` — new module: index build, upsert/remove/rename, path normalization
- `kb-mcp/src/server.rs` — wire shared index, lean `get_document`, refresh after `rebuild_index`
- `kb-mcp/src/watcher.rs` — keep document index in sync with incremental indexer events
- `kb-mcp/src/indexer.rs` — `pub(crate)` helpers reused by document index walk
- `kb-mcp/src/timing.rs` — add `response_build` to `GetDocumentStageTimingMs`
- `kb-mcp/tests/document_index_integration.rs` — MCP + latency + watcher refresh tests (`#[ignore]`)
- `docs/perf-instrumentation.md`, `README.md` — document direct-retrieval semantics

## Side-Effects Trace

- `get_document_from_index` / `get_document`: only readers of `document_index`; no `db` / `embedder` locks — search/rerank/vector paths untouched.
- `DocumentIndex::rebuild_from_kb`: called at startup and after `rebuild_index`; full scan duplicates disk reads already done by indexer during rebuild (acceptable, off hot path).
- Watcher `dispatch_*`: after DB indexer calls, upsert/remove/rename document index; failure logs to stderr, does not roll back SQLite index.
- `KbServerShared` / `WatcherState` gain `document_index` field — all `from_shared` / HTTP session clones share the same Arc (intended).
- Existing `validate_get_document_path` retained for `get_best_practice`; unit tests unchanged.
- New tests: `document_index` unit tests, `get_document_from_index` server tests (no embedder), integration tests ignored by default.

## Acceptance Criteria

- [x] In-memory index built at server startup for all readable KB docs
- [x] `get_document` reads index only; no search / embedding / vector / rerank
- [x] Index refreshed on full re-index and watcher events (create/modify/delete/rename)
- [x] Clear not-found when path missing from index
- [x] Existing response shape preserved; `cache_lookup`, `response_build`, `total` timings
- [x] Tests for index build, refresh, removal, not-found, no disk stages on hot path
- [x] Docs + latency integration test (`document_index_integration`)

## Test Plan

- Failing test first: `test_get_document_from_index_not_found` (path absent from index)
- Unit: `document_index` rebuild/upsert/remove/rename; `get_document_from_index` hit/miss
- Integration (`--ignored`): MCP `get_document` timing fields; median latency; watcher refresh
- Regression: `cargo test`, `cargo check`; existing server path-validation tests

## What I Am Most Likely Wrong About

- Startup memory footprint for very large vaults (full raw content held in RAM) may be unacceptable on constrained hosts; we assume typical agent KB sizes fit in memory and that skipping oversized files at index time is sufficient.
