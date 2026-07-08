## Diagnosis

- Root cause: MCP tool handlers returned JSON payloads with no structured latency breakdown; operators could not see whether slowness came from embedding, hybrid search, disk I/O, or serialization without external profilers.
- Confirmed by: `kb-mcp/src/server.rs` tool methods returning `serde_json::to_string_pretty(...)` with no timing fields; issue #1 body and ~700 ms unexplained `get_document` latency.
- Observability traps: end-to-end client latency includes rmcp transport overhead not visible inside handlers (`routing` stays 0); stage sums are indicative, not a strict partition.

## Proposed Fix

- Add `src/timing.rs` with monotonic `StageTimer` / `ToolRequestTimer` and tool-specific `timing_ms` structs.
- Instrument `search`, `get_document`, and `list_topics` at stage boundaries; attach top-level fields on success and error paths.
- Split hybrid search in `run_search_pipeline` when timing is requested so FTS, vector, RRF, reranker, and MMR are measured separately.
- Opt-out via `[instrumentation].timing_enabled = false` in `kb-mcp.toml`.

Alternatives considered: tracing-only (not visible in API responses); axum middleware (misses tool-internal stages).

## Files & Line Numbers

- `kb-mcp/src/timing.rs` — new module: timers, encode helpers, unit tests.
- `kb-mcp/src/config.rs` — `InstrumentationConfig` + `resolve_timing_enabled`.
- `kb-mcp/src/server.rs` — tool handlers, `run_search_pipeline` timing param, `KbServerShared.timing_enabled`.
- `kb-mcp/src/db.rs` — `merge_hybrid_rrf`, `pub(crate) search_fts_candidates`.
- `kb-mcp/src/main.rs` — pass `timing_enabled` into `run_server`.
- `docs/perf-instrumentation.md` — interpretation examples.
- `kb-mcp/kb-mcp.toml.example`, `README.md`, `docs/ARCHITECTURE.md` — config and docs.

## Side-Effects Trace

- `run_search_pipeline(..., pipeline_timing)`: CLI `search` and `eval` pass `None` — behavior unchanged. MCP `search` passes `Some` — extra vec+fts+RRF path when timing enabled (default); negligible vs search cost.
- `list_topics` response wraps array in `{ topics, timing_ms }` — clients parsing a bare array must read `.topics`; topic entry fields unchanged.
- `merge_hybrid_rrf` refactors existing RRF logic — same algorithm, shared by bounded/unbounded hybrid queries.
- `KbServerShared` gains `timing_enabled` — `for_test` defaults true; production from config.
- `build_document_response` retained for unit tests; `get_document` inlines parse timing in handler.

## Acceptance Criteria

- [x] `timing_ms` on MCP tool success/error responses (search, get_document, list_topics; generic on other tools via shared helpers where applicable)
- [x] Top-level fields: total, request_parse, routing, tool_execution, serialization
- [x] Search / get_document / list_topics stage fields per issue
- [x] `[instrumentation].timing_enabled` opt-out (default true)
- [x] No semantic change to existing result fields (results, path, content, etc.)
- [x] Tests for presence and internal consistency
- [x] Documentation with examples

## Test Plan

- Failing test first: `test_search_error_includes_consistent_timing_ms` in `server.rs` tests.
- `timing.rs` unit tests for timer arithmetic and encode helpers.
- Regression: `cargo test`, `cargo check`; existing search pipeline / db hybrid tests unchanged.

## What I Am Most Likely Wrong About

- `list_topics` wrapping `{ topics, ... }` may surprise clients that expected a top-level array; issue requires `timing_ms` on the response, which forced the wrapper. Reviewers should confirm downstream MCP clients tolerate the new envelope.
