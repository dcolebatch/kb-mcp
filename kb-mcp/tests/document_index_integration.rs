//! Integration tests for in-memory `get_document` direct retrieval (issue #2).
//!
//! `#[ignore]` because subprocess tests load BGE-small on first run.
//! Run: `cargo test --test document_index_integration -- --ignored`

mod common;
use common::mcp::{
    build_index, mcp_get_document_call, mcp_initialize, spawn_mcp_server,
    spawn_mcp_server_with_watch,
};
use common::temp::TempKbLayout;

use std::thread::sleep;
use std::time::{Duration, Instant};

fn setup_kb(layout: &TempKbLayout) {
    layout.write(
        "intro.md",
        "---\ntitle: Intro\ntopic: docs\n---\n\n## Overview\n\nDirect retrieval body.\n",
    );
}

#[test]
#[ignore = "spawns kb-mcp serve; needs BGE-small model download on first run"]
fn test_get_document_mcp_served_from_memory_index() {
    let layout = TempKbLayout::new("doc-index-mcp");
    setup_kb(&layout);
    build_index(layout.kb());

    let cfg_path = layout.root().join("kb-mcp.toml");
    std::fs::write(&cfg_path, "").expect("write kb-mcp.toml");

    let (_guard, base) = spawn_mcp_server(layout.kb(), &cfg_path);
    let session = mcp_initialize(&base);

    let resp = mcp_get_document_call(&base, &session, "intro.md");
    assert_eq!(resp["title"], "Intro");
    assert_eq!(resp["topic"], "docs");
    assert!(resp["content"].as_str().unwrap().contains("Direct retrieval"));
    let timing = &resp["timing_ms"];
    assert_eq!(timing["disk_read"], 0);
    assert_eq!(timing["frontmatter_parse"], 0);
    assert!(timing["cache_lookup"].as_u64().is_some());
    assert!(timing["response_build"].as_u64().is_some());
}

#[test]
#[ignore = "spawns kb-mcp serve; needs BGE-small model download on first run"]
fn test_get_document_latency_repeated_fetch() {
    let layout = TempKbLayout::new("doc-index-latency");
    setup_kb(&layout);
    build_index(layout.kb());

    let cfg_path = layout.root().join("kb-mcp.toml");
    std::fs::write(&cfg_path, "").expect("write kb-mcp.toml");

    let (_guard, base) = spawn_mcp_server(layout.kb(), &cfg_path);
    let session = mcp_initialize(&base);

    let mut samples = Vec::with_capacity(20);
    for _ in 0..20 {
        let start = Instant::now();
        let resp = mcp_get_document_call(&base, &session, "intro.md");
        samples.push(start.elapsed().as_millis() as u64);
        assert_eq!(resp["title"], "Intro");
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95) / 100];
    eprintln!("get_document latency ms: median={median} p95={p95} samples={samples:?}");
    assert!(
        median < 200,
        "median get_document latency should be well under search-class latency (got {median} ms)"
    );
}

#[test]
#[ignore = "spawns kb-mcp serve with watcher; needs inotify + BGE-small"]
fn test_get_document_watcher_refresh() {
    let layout = TempKbLayout::new("doc-index-watch");
    setup_kb(&layout);
    build_index(layout.kb());

    let cfg_path = layout.root().join("kb-mcp.toml");
    std::fs::write(
        &cfg_path,
        "[watch]\nenabled = true\ndebounce_ms = 500\n",
    )
    .expect("write kb-mcp.toml");

    let (_guard, base) = spawn_mcp_server_with_watch(layout.kb(), &cfg_path);
    let session = mcp_initialize(&base);
    sleep(Duration::from_millis(2000));

    layout.write(
        "fresh.md",
        "---\ntitle: Fresh\ntopic: watch\n---\n\n## New\n\nWatcher indexed.\n",
    );

    let deadline = Duration::from_secs(8);
    let start = Instant::now();
    loop {
        let resp = mcp_get_document_call(&base, &session, "fresh.md");
        if resp.get("title").and_then(|v| v.as_str()) == Some("Fresh") {
            break;
        }
        assert!(
            start.elapsed() < deadline,
            "watcher did not refresh document index for fresh.md"
        );
        sleep(Duration::from_millis(250));
    }

    std::fs::remove_file(layout.kb().join("fresh.md")).expect("remove fresh.md");
    sleep(Duration::from_millis(2000));
    let gone = mcp_get_document_call(&base, &session, "fresh.md");
    assert!(gone.get("error").is_some(), "deleted file should not be in index");
}
