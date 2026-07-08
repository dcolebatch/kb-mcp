//! In-memory document index for deterministic `get_document(path)` retrieval.
//!
//! Built at server startup and kept in sync via `rebuild_index` and the file
//! watcher. Hot-path lookups avoid disk I/O and the search / embedding pipeline.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};

use crate::indexer::{collect_source_files, extract_category_topic, sha256_hex};
use crate::parser::Registry;

/// Maximum document size indexed and served via `get_document`.
pub const GET_DOCUMENT_MAX_BYTES: u64 = 1024 * 1024;

/// One indexed knowledge-base document keyed by canonical relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentEntry {
    /// Canonical relative path (forward slashes).
    pub path: String,
    pub title: Option<String>,
    pub date: Option<String>,
    pub topic: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    /// Full raw file content (Markdown / text as stored on disk).
    pub content: String,
    /// Parsed body text when chunking produced content (joined chunk bodies).
    pub body: Option<String>,
    pub content_hash: String,
    pub last_modified: Option<SystemTime>,
}

/// Thread-safe in-memory map from canonical relative path → document entry.
#[derive(Debug, Clone, Default)]
pub struct DocumentIndex {
    docs: HashMap<String, DocumentEntry>,
}

impl DocumentIndex {
    pub fn new() -> Self {
        Self {
            docs: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn contains(&self, rel_path: &str) -> bool {
        self.docs.contains_key(rel_path)
    }

    pub fn get(&self, rel_path: &str) -> Option<&DocumentEntry> {
        self.docs.get(rel_path)
    }

    pub fn remove(&mut self, rel_path: &str) -> bool {
        self.docs.remove(rel_path).is_some()
    }

    pub fn rename(&mut self, old_rel: &str, new_rel: &str) -> bool {
        let Some(mut entry) = self.docs.remove(old_rel) else {
            return false;
        };
        entry.path = new_rel.to_string();
        self.docs.insert(new_rel.to_string(), entry);
        true
    }

    /// Full scan of readable KB files. Used at server startup and after
    /// `rebuild_index`.
    pub fn rebuild_from_kb(
        &mut self,
        kb_path: &Path,
        exclude_dirs: &[String],
        registry: &Registry,
        exclude_headings: Option<&[String]>,
        max_bytes: u64,
    ) -> Result<usize> {
        let kb_path = kb_path
            .canonicalize()
            .with_context(|| format!("failed to canonicalize kb_path: {}", kb_path.display()))?;
        let files = collect_source_files(&kb_path, registry, exclude_dirs)?;
        self.docs.clear();
        let mut loaded = 0usize;
        for full in files {
            let rel = rel_path_from_full(&kb_path, &full);
            if let Some(entry) =
                load_entry_from_file(&kb_path, &full, &rel, registry, exclude_headings, max_bytes)?
            {
                self.docs.insert(rel, entry);
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Incremental upsert after watcher reindex or manual file change.
    pub fn upsert_from_rel(
        &mut self,
        kb_path: &Path,
        rel: &str,
        registry: &Registry,
        exclude_headings: Option<&[String]>,
        max_bytes: u64,
    ) -> Result<bool> {
        let full = kb_path.join(rel);
        if !full.exists() {
            return Ok(self.remove(rel));
        }
        if let Some(entry) =
            load_entry_from_file(kb_path, &full, rel, registry, exclude_headings, max_bytes)?
        {
            self.docs.insert(rel.to_string(), entry);
            Ok(true)
        } else {
            Ok(self.remove(rel))
        }
    }
}

pub type SharedDocumentIndex = Arc<RwLock<DocumentIndex>>;

pub fn new_shared_index() -> SharedDocumentIndex {
    Arc::new(RwLock::new(DocumentIndex::new()))
}

/// Normalize a client-supplied relative path to the canonical index key.
pub fn normalize_rel_path(rel_path: &str) -> Option<String> {
    let trimmed = rel_path.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') || trimmed.contains('\\') {
        return None;
    }
    if trimmed.split('/').any(|p| p == ".." || p.is_empty()) {
        return None;
    }
    Some(trimmed.to_string())
}

/// Lightweight path-key validation (no disk I/O). Mirrors extension rules used
/// by the search index without stat/read syscalls.
pub fn validate_rel_path_key(rel_path: &str, registry: &Registry) -> Result<(), String> {
    let Some(normalized) = normalize_rel_path(rel_path) else {
        return Err(format!(
            "File not found: {rel_path}. Path should be relative to knowledge-base/ (e.g. \"deep-dive/mcp/overview.md\")."
        ));
    };
    let ext = Path::new(&normalized)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if !registry.has_extension(ext) {
        return Err(format!(
            "Access denied: extension {ext:?} is not in the indexed parser registry. Allowed: {:?}",
            registry.extensions()
        ));
    }
    Ok(())
}

fn rel_path_from_full(kb_path: &Path, full: &Path) -> String {
    full.strip_prefix(kb_path)
        .unwrap_or(full)
        .to_string_lossy()
        .replace('\\', "/")
}

fn load_entry_from_file(
    kb_path: &Path,
    full: &Path,
    rel: &str,
    registry: &Registry,
    exclude_headings: Option<&[String]>,
    max_bytes: u64,
) -> Result<Option<DocumentEntry>> {
    if std::fs::symlink_metadata(full)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(true)
    {
        return Ok(None);
    }
    let canonical = match full.canonicalize() {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    if !canonical.starts_with(kb_path) {
        return Ok(None);
    }
    let meta = match std::fs::metadata(&canonical) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if meta.len() > max_bytes {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&canonical)
        .with_context(|| format!("failed to read {}", canonical.display()))?;
    let ext = canonical
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if !registry.has_extension(ext) {
        return Ok(None);
    }
    let excludes: Vec<&str> = match exclude_headings {
        Some(list) => list.iter().map(String::as_str).collect(),
        None => crate::parser::DEFAULT_EXCLUDED_HEADINGS.to_vec(),
    };
    let parsed = match registry.by_extension(ext) {
        Some(p) => p.parse(&content, rel, &excludes),
        None => return Ok(None),
    };
    if parsed.chunks.is_empty() {
        return Ok(None);
    }
    let (category, path_topic) = extract_category_topic(rel);
    let topic = parsed
        .frontmatter
        .topic
        .clone()
        .or(path_topic);
    let body = if parsed.chunks.is_empty() {
        None
    } else {
        Some(
            parsed
                .chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    };
    let hash = sha256_hex(&content);
    Ok(Some(DocumentEntry {
        path: rel.to_string(),
        title: parsed.frontmatter.title,
        date: parsed.frontmatter.date,
        topic,
        category,
        tags: parsed.frontmatter.tags,
        content,
        body,
        content_hash: hash,
        last_modified: meta.modified().ok(),
    }))
}

/// Build a shared index at server startup (logs count to stderr).
pub fn build_shared_at_startup(
    kb_path: &Path,
    exclude_dirs: &[String],
    registry: &Registry,
    exclude_headings: Option<&[String]>,
    max_bytes: u64,
) -> Result<SharedDocumentIndex> {
    let index = new_shared_index();
    {
        let mut guard = index
            .write()
            .map_err(|_| anyhow::anyhow!("document index lock poisoned"))?;
        let n = guard.rebuild_from_kb(kb_path, exclude_dirs, registry, exclude_headings, max_bytes)?;
        eprintln!("Document index: {n} files loaded into memory");
    }
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::parser::Registry;

    struct TempKb {
        path: PathBuf,
    }

    impl TempKb {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "kb-mcp-doc-index-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn write(&self, rel: &str, content: &str) {
            let full = self.path.join(rel);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(full, content).unwrap();
        }
    }

    impl Drop for TempKb {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    const SAMPLE: &str = r#"---
title: Hello
date: 2026-01-01
topic: mcp
tags: [a, b]
---

# Body

Content here.
"#;

    #[test]
    fn test_normalize_rel_path_rejects_traversal() {
        assert!(normalize_rel_path("../secret.md").is_none());
        assert!(normalize_rel_path("docs//a.md").is_none());
        assert_eq!(
            normalize_rel_path("docs/a.md").as_deref(),
            Some("docs/a.md")
        );
    }

    #[test]
    fn test_rebuild_from_kb_indexes_readable_docs() {
        let kb = TempKb::new();
        kb.write("notes/a.md", SAMPLE);
        let mut index = DocumentIndex::new();
        let registry = Registry::default();
        let n = index
            .rebuild_from_kb(&kb.path, &[], &registry, None, 1024 * 1024)
            .unwrap();
        assert_eq!(n, 1);
        let entry = index.get("notes/a.md").expect("indexed");
        assert_eq!(entry.title.as_deref(), Some("Hello"));
        assert_eq!(entry.topic.as_deref(), Some("mcp"));
        assert_eq!(entry.tags, vec!["a", "b"]);
        assert!(entry.content.contains("# Body"));
        assert!(entry.body.as_ref().is_some_and(|b| b.contains("Content here")));
        assert!(!entry.content_hash.is_empty());
    }

    #[test]
    fn test_upsert_and_remove_on_change() {
        let kb = TempKb::new();
        kb.write("a.md", SAMPLE);
        let registry = Registry::default();
        let mut index = DocumentIndex::new();
        index
            .rebuild_from_kb(&kb.path, &[], &registry, None, 1024 * 1024)
            .unwrap();
        assert!(index.contains("a.md"));

        let updated = r#"---
title: Updated
---

New body.
"#;
        kb.write("a.md", updated);
        index
            .upsert_from_rel(&kb.path, "a.md", &registry, None, 1024 * 1024)
            .unwrap();
        assert_eq!(
            index.get("a.md").unwrap().title.as_deref(),
            Some("Updated")
        );

        fs::remove_file(kb.path.join("a.md")).unwrap();
        index
            .upsert_from_rel(&kb.path, "a.md", &registry, None, 1024 * 1024)
            .unwrap();
        assert!(!index.contains("a.md"));
    }

    #[test]
    fn test_rename_updates_both_paths() {
        let kb = TempKb::new();
        kb.write("old.md", SAMPLE);
        let registry = Registry::default();
        let mut index = DocumentIndex::new();
        index
            .rebuild_from_kb(&kb.path, &[], &registry, None, 1024 * 1024)
            .unwrap();
        kb.write("new.md", SAMPLE);
        fs::remove_file(kb.path.join("old.md")).unwrap();
        index.remove("old.md");
        index
            .upsert_from_rel(&kb.path, "new.md", &registry, None, 1024 * 1024)
            .unwrap();
        assert!(!index.contains("old.md"));
        assert!(index.contains("new.md"));
    }

    #[test]
    fn test_not_found_when_missing() {
        let index = DocumentIndex::new();
        assert!(index.get("missing.md").is_none());
    }
}
