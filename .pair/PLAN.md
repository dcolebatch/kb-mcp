## Diagnosis

- Root cause: `Database::list_topics()` aggregates only `title` via `json_group_array(title)` and never returns `documents.path`, the vault-relative key that `get_document(path)` requires. Agents see titles but cannot deterministically call `get_document` without guessing paths or falling back to `search()`.
- Confirmed by: `kb-mcp/src/db.rs` `TopicInfo` (fields: category/topic/file_count/last_updated/titles only); `list_topics` SQL omits `path`; `kb-mcp/src/server.rs` `TopicEntry` mirrors that shape; issue #5 reproduction steps.
- Observability traps: `file_count` and `titles` look complete and healthy while the navigation workflow is still broken for agents.

## Proposed Fix

- Extend `TopicInfo` / MCP `TopicEntry` with `documents: [{ title, path }]` sourced from the same `documents` rows, keep existing fields for backwards compatibility, assert `file_count == documents.len()`.
- Why this and not (1) embedding paths into `titles` strings — would break BC and force parsing; (2) a separate `list_documents` tool — adds a hop agents must discover and does not fix `list_topics` itself.

## Files & Line Numbers

- `kb-mcp/src/db.rs` — add `TopicDocument`, extend `TopicInfo`, extend `list_topics()` SQL/`json_object` parse
- `kb-mcp/src/server.rs` — `TopicEntry` + tool description + mapping
- `kb-mcp/src/db.rs` tests — shape, BC fields, path → `get_document_hash` usability
- `README.md` / `README.ja.md` — MCP tool table + recommended workflow

## Side-Effects Trace

- `Database::list_topics()` callers: MCP `server::list_topics` only (plus unit tests). CLI has no separate topics dump of this struct. Adding a field is additive JSON; existing consumers that ignore unknown fields stay valid.
- `TopicInfo` / `TopicEntry`: new field only; `titles` aggregation logic unchanged (still flattens null titles), so `titles.len()` may remain `< file_count` when some titles are NULL — pre-existing.
- Shared state: read-only SELECT on `documents`; no schema migration, no index rebuild, no cache invalidation.
- New path: `documents` array paths must equal stored `documents.path` (same key as document index / `get_document`); covered by new test via `get_document_hash(path)`.
- Uncovered initially: MCP HTTP integration for `list_topics` JSON — unit-level DB + server mapping is sufficient for AC; no HTTP golden for topics today.

## Acceptance Criteria

- [x] `list_topics()` retains `category`, `topic`, `file_count`, `last_updated`, `titles`
- [x] Each topic entry includes `documents: [{ title, path }]`
- [x] Every `path` is usable as `get_document(path)` without transformation
- [x] `file_count == documents.len()`
- [x] Automated tests cover BC fields + `documents`
- [x] README.md / README.ja.md document the field and `list_topics` → `get_document(path)` workflow

## Test Plan

- Failing test first: `test_list_topics_documents_include_paths_usable_by_get_document` in `kb-mcp/src/db.rs` — asserts `documents` shape, `file_count` consistency, and `get_document_hash(path).is_some()` for each path
- Extend `test_list_topics` for `documents` presence alongside existing title assertions
- Regression: `test_list_topics_title_with_double_pipe_is_not_split`, full `cargo test`

## What I Am Most Likely Wrong About

- Whether `title` inside `documents` should be `Option<String>` (honest DB nulls) vs always a string (empty when missing). AC says each element has `title` and `path`; null JSON for title preserves document rows with `file_count` parity better than dropping them the way `titles` does.
