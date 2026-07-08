//! Lightweight monotonic timing helpers for MCP tool responses.
//!
//! Stage timers use [`std::time::Instant`] (monotonic, no syscall per read beyond
//! `elapsed()`). Overhead is well under 1 ms per request when enabled.

use std::time::Instant;

use serde::Serialize;

/// Convert an elapsed duration to whole milliseconds (rounded).
#[inline]
pub fn instant_to_ms(start: Instant) -> u64 {
    (start.elapsed().as_micros().div_ceil(1000)) as u64
}

/// Monotonic stage timer. Cheap to create; records elapsed ms on demand.
#[derive(Debug)]
pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    #[inline]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        instant_to_ms(self.start)
    }
}

/// Top-level timing fields present on every MCP tool response.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct BaseTimingMs {
    pub total: u64,
    pub request_parse: u64,
    pub routing: u64,
    pub tool_execution: u64,
    pub serialization: u64,
}

/// Tracks wall-clock phases for a single MCP tool handler invocation.
#[derive(Debug)]
pub struct ToolRequestTimer {
    start: Instant,
    parse_done: Option<Instant>,
}

impl ToolRequestTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            parse_done: None,
        }
    }

    /// Call after parameter validation / early rejects setup completes.
    pub fn mark_parse_done(&mut self) {
        self.parse_done = Some(Instant::now());
    }

    /// Build top-level timing. `routing` is always 0 — rmcp dispatches to the
    /// tool handler before our code runs, so routing latency is not observable
    /// inside the handler.
    pub fn finish(&self, serialization_ms: u64) -> BaseTimingMs {
        let total = instant_to_ms(self.start);
        let request_parse = self
            .parse_done
            .map(|p| (p.duration_since(self.start).as_micros().div_ceil(1000)) as u64)
            .unwrap_or(0);
        let routing = 0u64;
        let serialization = serialization_ms;
        let tool_execution = total
            .saturating_sub(request_parse)
            .saturating_sub(routing)
            .saturating_sub(serialization);
        BaseTimingMs {
            total,
            request_parse,
            routing,
            tool_execution,
            serialization,
        }
    }
}

/// Per-stage breakdown for the `search` MCP tool (nested under `timing_ms`).
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct SearchStageTimingMs {
    #[serde(flatten)]
    pub base: BaseTimingMs,
    pub embedding_generation: u64,
    pub sqlite_fts: u64,
    pub vector_search: u64,
    pub reciprocal_rank_fusion: u64,
    pub reranker: u64,
    pub mmr: u64,
    pub parent_retriever: u64,
    pub result_filtering: u64,
    pub response_build: u64,
}

/// Per-stage breakdown for the `get_document` MCP tool.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct GetDocumentStageTimingMs {
    #[serde(flatten)]
    pub base: BaseTimingMs,
    pub document_lookup: u64,
    pub cache_lookup: u64,
    pub disk_read: u64,
    pub frontmatter_parse: u64,
    pub markdown_load: u64,
    pub response_build: u64,
}

/// Per-stage breakdown for the `list_topics` MCP tool.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct ListTopicsStageTimingMs {
    #[serde(flatten)]
    pub base: BaseTimingMs,
    pub topic_index_lookup: u64,
    pub response_build: u64,
}

/// Generic timing wrapper for tools without a dedicated stage breakdown.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct GenericToolTimingMs {
    #[serde(flatten)]
    pub base: BaseTimingMs,
}

/// Timings collected inside [`crate::server::run_search_pipeline`].
#[derive(Debug, Clone, Default)]
pub struct SearchPipelineTiming {
    pub sqlite_fts: u64,
    pub vector_search: u64,
    pub reciprocal_rank_fusion: u64,
    pub reranker: u64,
    pub mmr: u64,
}

/// Serialize `body` to pretty JSON and attach `timing_ms` when enabled.
pub fn encode_with_timing<T: Serialize>(
    timing_enabled: bool,
    body: &T,
    timing: impl Serialize,
) -> String {
    if !timing_enabled {
        return serde_json::to_string_pretty(body).unwrap_or_default();
    }
    let mut value = serde_json::to_value(body).unwrap_or(serde_json::Value::Null);
    if let serde_json::Value::Object(ref mut map) = value {
        map.insert(
            "timing_ms".to_string(),
            serde_json::to_value(timing).unwrap_or(serde_json::Value::Null),
        );
    }
    let ser_timer = StageTimer::start();
    let out = serde_json::to_string_pretty(&value).unwrap_or_default();
    // Serialization time for the outer envelope is accounted by callers via
    // `ToolRequestTimer::finish(ser_ms)`; this helper only builds the payload.
    let _ = ser_timer;
    out
}

/// Serialize an error payload with optional `timing_ms`.
pub fn encode_error_with_timing(
    timing_enabled: bool,
    error: &str,
    timing: impl Serialize,
) -> String {
    if !timing_enabled {
        return serde_json::to_string_pretty(&serde_json::json!({ "error": error }))
            .unwrap_or_default();
    }
    let mut value = serde_json::json!({ "error": error });
    if let Some(map) = value.as_object_mut() {
        map.insert(
            "timing_ms".to_string(),
            serde_json::to_value(timing).unwrap_or(serde_json::Value::Null),
        );
    }
    serde_json::to_string_pretty(&value).unwrap_or_default()
}

/// Verify stage timings are internally consistent with `total`.
pub fn stages_sum_reasonable(total: u64, stages: &[u64]) -> bool {
    let sum: u64 = stages.iter().sum();
    // Stages may overlap with base buckets or include idle; allow generous slack.
    sum <= total.saturating_add(5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_request_timer_finish_accounts_for_phases() {
        let mut timer = ToolRequestTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(2));
        timer.mark_parse_done();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let base = timer.finish(1);
        assert!(base.total >= 4);
        assert!(base.request_parse >= 2);
        assert_eq!(base.routing, 0);
        assert_eq!(base.serialization, 1);
        assert_eq!(
            base.total,
            base.request_parse + base.routing + base.tool_execution + base.serialization
        );
    }

    #[test]
    fn test_encode_with_timing_adds_field() {
        #[derive(Serialize)]
        struct Body {
            ok: bool,
        }
        let out = encode_with_timing(
            true,
            &Body { ok: true },
            BaseTimingMs {
                total: 10,
                request_parse: 1,
                routing: 0,
                tool_execution: 8,
                serialization: 1,
            },
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["timing_ms"]["total"], 10);
    }

    #[test]
    fn test_encode_with_timing_disabled_omits_field() {
        #[derive(Serialize)]
        struct Body {
            ok: bool,
        }
        let out = encode_with_timing(
            false,
            &Body { ok: true },
            BaseTimingMs::default(),
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("timing_ms").is_none());
    }

    #[test]
    fn test_stages_sum_reasonable() {
        assert!(stages_sum_reasonable(100, &[30, 40, 20]));
        assert!(!stages_sum_reasonable(10, &[20, 20]));
    }
}
