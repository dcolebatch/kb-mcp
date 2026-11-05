use anyhow::Result;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
use serde::Deserialize;
use std::path::PathBuf;

use crate::db::SearchResult;

/// Embedding モデル選択肢。CLI の `--model` と共有される。
///
/// 追加時の手順: variant を足し、`model_id` / `dimension` /
/// `fastembed_model` / `approx_download_mb` の 4 メソッドに分岐を追加する。
///
/// デフォルトは既存 DB 互換のため `BgeSmallEnV15` に固定 (`#[default]`)。
/// BGE-M3 へ切り替えたい場合は CLI で明示オプトインする。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum, Deserialize)]
pub enum ModelChoice {
    /// BAAI/bge-small-en-v1.5 (384 dim, 英語特化, ~130 MB)
    #[default]
    #[value(name = "bge-small-en-v1.5")]
    #[serde(rename = "bge-small-en-v1.5")]
    BgeSmallEnV15,
    /// BAAI/bge-m3 (1024 dim, 多言語, ~2.3 GB)
    #[value(name = "bge-m3")]
    #[serde(rename = "bge-m3")]
    BgeM3,
}

impl ModelChoice {
    pub fn model_id(self) -> &'static str {
        match self {
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::BgeM3 => "bge-m3",
        }
    }

    pub fn dimension(self) -> usize {
        match self {
            Self::BgeSmallEnV15 => 384,
            Self::BgeM3 => 1024,
        }
    }

    fn fastembed_model(self) -> EmbeddingModel {
        match self {
            Self::BgeSmallEnV15 => EmbeddingModel::BGESmallENV15,
            Self::BgeM3 => EmbeddingModel::BGEM3,
        }
    }

    /// 初回 DL サイズの目安 (ユーザ告知用)
    fn approx_download_mb(self) -> u32 {
        match self {
            Self::BgeSmallEnV15 => 130,
            Self::BgeM3 => 2300,
        }
    }

    /// fastembed の `embed()` に渡すバッチサイズ。モデルの 1 トークンあたりの
    /// activation memory が違うため、大きなモデルでは小さめのバッチに絞って
    /// OOM を避ける。
    ///
    /// 計算根拠: `batch * max_length(=512) * hidden_dim * 4 bytes`
    /// - BgeSmallEnV15 (384 dim) @ 256 → ~200 MB
    /// - BgeM3         (1024 dim) @ 32 → ~67 MB
    pub fn batch_size(self) -> usize {
        match self {
            Self::BgeSmallEnV15 => 256,
            Self::BgeM3 => 32,
        }
    }
}

/// Resolved embedding backend + the index identity (`model_id`, dimension).
///
/// Built from config / CLI before contacting any remote endpoint so
/// `verify_embedding_meta` stays deterministic.
#[derive(Clone, Debug)]
pub struct EmbeddingSettings {
    provider: ProviderSettings,
    model_id: String,
    dimension: usize,
}

#[derive(Clone, Debug)]
enum ProviderSettings {
    FastEmbed(ModelChoice),
    OpenAiCompatible(OpenAiCompatibleConfig),
}

/// Config for an OpenAI-compatible `POST /v1/embeddings` endpoint (Ollama, TEI, …).
#[derive(Clone, Debug)]
pub struct OpenAiCompatibleConfig {
    pub endpoint: String,
    pub model: String,
    pub dimension: usize,
    pub timeout_seconds: u64,
    pub batch_size: usize,
    pub api_key: Option<String>,
}

impl OpenAiCompatibleConfig {
    /// Index-meta identity: `openai:{alias}`.
    pub fn model_id(model: &str) -> String {
        format!("openai:{model}")
    }

    pub fn validate(&self) -> Result<()> {
        if self.endpoint.trim().is_empty() {
            anyhow::bail!("[embedding].endpoint must be a non-empty URL");
        }
        if self.model.trim().is_empty() {
            anyhow::bail!("[embedding].model must be a non-empty alias");
        }
        if self.dimension == 0 {
            anyhow::bail!("[embedding].dimension must be > 0");
        }
        if self.timeout_seconds == 0 {
            anyhow::bail!("[embedding].timeout_seconds must be > 0");
        }
        if self.batch_size == 0 {
            anyhow::bail!("[embedding].batch_size must be > 0");
        }
        let url = reqwest::Url::parse(self.endpoint.trim())
            .map_err(|e| anyhow::anyhow!("[embedding].endpoint is not a valid URL: {e}"))?;
        if url.scheme() != "http" && url.scheme() != "https" {
            anyhow::bail!(
                "[embedding].endpoint must use http or https, got {}",
                url.scheme()
            );
        }
        if !url.username().is_empty() || url.password().is_some() {
            anyhow::bail!(
                "[embedding].endpoint must not embed credentials; use [embedding].api_key instead"
            );
        }
        Ok(())
    }
}

impl EmbeddingSettings {
    pub fn fastembed(choice: ModelChoice) -> Self {
        Self {
            provider: ProviderSettings::FastEmbed(choice),
            model_id: choice.model_id().to_string(),
            dimension: choice.dimension(),
        }
    }

    pub fn openai_compatible(config: OpenAiCompatibleConfig) -> Result<Self> {
        config.validate()?;
        let model_id = OpenAiCompatibleConfig::model_id(config.model.trim());
        let dimension = config.dimension;
        Ok(Self {
            provider: ProviderSettings::OpenAiCompatible(config),
            model_id,
            dimension,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn is_openai_compatible(&self) -> bool {
        matches!(self.provider, ProviderSettings::OpenAiCompatible(_))
    }
}

enum Backend {
    FastEmbed {
        model: TextEmbedding,
        choice: ModelChoice,
    },
    OpenAi(OpenAiHttpBackend),
}

struct OpenAiHttpBackend {
    client: reqwest::blocking::Client,
    endpoint: String,
    model: String,
    dimension: usize,
    batch_size: usize,
}

/// Thin wrapper around FastEmbed or an OpenAI-compatible HTTP embedder.
pub struct Embedder {
    backend: Backend,
    model_id: String,
    dimension: usize,
}

impl Embedder {
    /// デフォルトモデル ([`ModelChoice::default`]) で初期化する。
    ///
    /// Cache directory resolution (in order):
    /// 1. `FASTEMBED_CACHE_DIR` environment variable if set
    /// 2. OS-standard cache directory joined with `fastembed`
    ///    (Linux: `~/.cache/fastembed`, macOS: `~/Library/Caches/fastembed`,
    ///    Windows: `%LOCALAPPDATA%\fastembed`)
    /// 3. `.fastembed_cache` relative to the working directory (fastembed's own default)
    pub fn new() -> Result<Self> {
        Self::with_model(ModelChoice::default())
    }

    /// 明示的に FastEmbed モデルを指定して初期化する。
    pub fn with_model(choice: ModelChoice) -> Result<Self> {
        Self::with_settings(EmbeddingSettings::fastembed(choice))
    }

    /// Initialize from resolved config / CLI settings.
    pub fn with_settings(settings: EmbeddingSettings) -> Result<Self> {
        let EmbeddingSettings {
            provider,
            model_id,
            dimension,
        } = settings;
        let backend = match provider {
            ProviderSettings::FastEmbed(choice) => {
                eprintln!(
                    "Loading embedding model: {} ({} dim, ~{} MB on first run)...",
                    choice.model_id(),
                    choice.dimension(),
                    choice.approx_download_mb()
                );
                let model = TextEmbedding::try_new(
                    InitOptions::new(choice.fastembed_model())
                        .with_cache_dir(resolve_cache_dir())
                        .with_show_download_progress(true),
                )?;
                Backend::FastEmbed { model, choice }
            }
            ProviderSettings::OpenAiCompatible(config) => {
                eprintln!(
                    "Using OpenAI-compatible embeddings: {} ({} dim) via {}",
                    config.model.trim(),
                    config.dimension,
                    config.endpoint.trim()
                );
                let mut builder = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(config.timeout_seconds))
                    .redirect(reqwest::redirect::Policy::none());
                if let Some(key) = config.api_key.as_ref().filter(|k| !k.is_empty()) {
                    let mut headers = reqwest::header::HeaderMap::new();
                    let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                        .map_err(|e| anyhow::anyhow!("invalid [embedding].api_key for Authorization header: {e}"))?;
                    headers.insert(reqwest::header::AUTHORIZATION, value);
                    builder = builder.default_headers(headers);
                }
                let client = builder
                    .build()
                    .map_err(|e| anyhow::anyhow!("failed to build HTTP embedding client: {e}"))?;
                Backend::OpenAi(OpenAiHttpBackend {
                    client,
                    endpoint: config.endpoint.trim().to_string(),
                    model: config.model.trim().to_string(),
                    dimension: config.dimension,
                    batch_size: config.batch_size,
                })
            }
        };
        Ok(Self {
            backend,
            model_id,
            dimension,
        })
    }

    /// Embed multiple texts in a batch.
    pub fn embed_texts(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &mut self.backend {
            Backend::FastEmbed { model, choice } => {
                let embeddings = model.embed(texts, Some(choice.batch_size()))?;
                Ok(embeddings)
            }
            Backend::OpenAi(http) => http.embed_texts(texts),
        }
    }

    /// Embed a single text.
    pub fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_texts(&[text])?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding returned empty result"))
    }

    /// 選択中のモデルの埋め込み次元数。
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 選択中のモデルの識別子 (index_meta に記録される)。
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// True when this embedder uses a blocking HTTP client (must not run on a
    /// tokio worker without `spawn_blocking`).
    pub fn uses_blocking_http(&self) -> bool {
        matches!(self.backend, Backend::OpenAi(_))
    }
}

impl OpenAiHttpBackend {
    fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let batch = self.embed_batch(chunk)?;
            out.extend(batch);
        }
        Ok(out)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        #[derive(serde::Serialize)]
        struct Request<'a> {
            model: &'a str,
            input: &'a [&'a str],
        }
        #[derive(serde::Deserialize)]
        struct Response {
            data: Vec<Item>,
        }
        #[derive(serde::Deserialize)]
        struct Item {
            index: Option<usize>,
            embedding: Vec<f32>,
        }

        let body = Request {
            model: &self.model,
            input: texts,
        };
        let response = self
            .client
            .post(&self.endpoint)
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("embedding HTTP request failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            let snippet = response
                .text()
                .unwrap_or_default()
                .chars()
                .take(512)
                .collect::<String>();
            anyhow::bail!(
                "embedding endpoint returned HTTP {status}: {}",
                snippet.replace('\n', " ")
            );
        }
        let parsed: Response = response
            .json()
            .map_err(|e| anyhow::anyhow!("embedding response JSON parse failed: {e}"))?;
        if parsed.data.len() != texts.len() {
            anyhow::bail!(
                "embedding response length mismatch: got {} vectors for {} inputs",
                parsed.data.len(),
                texts.len()
            );
        }

        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for (fallback_i, item) in parsed.data.into_iter().enumerate() {
            let idx = item.index.unwrap_or(fallback_i);
            if idx >= texts.len() {
                anyhow::bail!("embedding response index {idx} out of range for {} inputs", texts.len());
            }
            if ordered[idx].is_some() {
                anyhow::bail!("embedding response has duplicate index {idx}");
            }
            if item.embedding.len() != self.dimension {
                anyhow::bail!(
                    "embedding dimension mismatch at index {idx}: expected {}, got {}",
                    self.dimension,
                    item.embedding.len()
                );
            }
            ordered[idx] = Some(item.embedding);
        }
        ordered
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.ok_or_else(|| anyhow::anyhow!("embedding response missing index {i}"))
            })
            .collect()
    }
}

fn resolve_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FASTEMBED_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(base) = dirs::cache_dir() {
        return base.join("fastembed");
    }
    PathBuf::from(".fastembed_cache")
}

// ---------------------------------------------------------------------------
// Reranker
// ---------------------------------------------------------------------------

/// Cross-encoder reranker の選択肢。CLI `--reranker` と共有される。
///
/// デフォルトは `None` (reranker 無効)。モデル DL を避けるため、opt-in で
/// 明示的に選択する。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum, Deserialize)]
pub enum RerankerChoice {
    /// reranker 無効 (RRF 結果をそのまま返す)
    #[default]
    #[value(name = "none")]
    #[serde(rename = "none")]
    None,
    /// BAAI/bge-reranker-v2-m3 (多言語 100+ 言語, ~2.3 GB)。日本語 KB 推奨
    #[value(name = "bge-v2-m3")]
    #[serde(rename = "bge-v2-m3")]
    BgeV2M3,
    /// jinaai/jina-reranker-v2-base-multilingual (多言語, ~1.2 GB)。軽量多言語
    #[value(name = "jina-v2-ml")]
    #[serde(rename = "jina-v2-ml")]
    JinaV2Multilingual,
    /// BAAI/bge-reranker-base (英/中のみ, ~280 MB)。日本語用途には非推奨
    #[value(name = "bge-base")]
    #[serde(rename = "bge-base")]
    BgeBase,
}

impl RerankerChoice {
    pub fn model_id(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BgeV2M3 => "bge-reranker-v2-m3",
            Self::JinaV2Multilingual => "jina-reranker-v2-base-multilingual",
            Self::BgeBase => "bge-reranker-base",
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    fn fastembed_model(self) -> Option<RerankerModel> {
        match self {
            Self::None => None,
            Self::BgeV2M3 => Some(RerankerModel::BGERerankerV2M3),
            Self::JinaV2Multilingual => Some(RerankerModel::JINARerankerV2BaseMultiligual),
            Self::BgeBase => Some(RerankerModel::BGERerankerBase),
        }
    }

    fn approx_download_mb(self) -> u32 {
        match self {
            Self::None => 0,
            Self::BgeV2M3 => 2300,
            Self::JinaV2Multilingual => 1200,
            Self::BgeBase => 280,
        }
    }
}

/// Cross-encoder reranker。`search_hybrid` が返した候補を query との共同
/// エンコードで再スコア付けし、上位 `limit` 件に絞る。
pub struct Reranker {
    model: TextRerank,
    #[allow(dead_code)] // choice は model_id ログ用に保持
    choice: RerankerChoice,
}

impl Reranker {
    /// `choice == None` のときは `Ok(None)` を返す (DL・ロード共にスキップ)。
    /// それ以外は ONNX モデルをロードし `Some(Reranker)` を返す。
    pub fn try_new(choice: RerankerChoice) -> Result<Option<Self>> {
        let Some(fm) = choice.fastembed_model() else {
            return Ok(None);
        };
        eprintln!(
            "Loading reranker model: {} (~{} MB on first run)...",
            choice.model_id(),
            choice.approx_download_mb()
        );
        let model = TextRerank::try_new(
            RerankInitOptions::new(fm)
                .with_cache_dir(resolve_cache_dir())
                .with_show_download_progress(true),
        )?;
        Ok(Some(Self { model, choice }))
    }

    /// `candidates` (chunk_id, SearchResult) を cross-encoder でスコア付けし、
    /// 降順にソートした上位 `limit` 件の `SearchResult` を返す。
    /// `score` フィールドには reranker の raw score が入る (大きいほど良い)。
    pub fn rerank_candidates(
        &mut self,
        query: &str,
        candidates: Vec<(i64, SearchResult)>,
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .rerank_candidates_with_ids(query, candidates, limit)?
            .into_iter()
            .map(|(_, r)| r)
            .collect())
    }

    /// `rerank_candidates` と同じ結果を `(chunk_id, SearchResult)` で返す版。
    /// MMR の relevance 入力に chunk_id を保持したまま渡したいユースケース
    /// (feature-28 Task 2.9) で使う。`rerank_candidates` はこれに委譲する形に
    /// なっており、挙動は完全一致 (score / 順序とも同一)。
    pub fn rerank_candidates_with_ids(
        &mut self,
        query: &str,
        candidates: Vec<(i64, SearchResult)>,
        limit: u32,
    ) -> Result<Vec<(i64, SearchResult)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let documents: Vec<&str> = candidates.iter().map(|(_, r)| r.content.as_str()).collect();
        let rerank_results = self.model.rerank(query, documents, false, None)?;

        // rerank_results は score 降順でソート済み。index は documents (= candidates) の位置。
        let mut out: Vec<(i64, SearchResult)> = Vec::with_capacity(limit as usize);
        for r in rerank_results.into_iter().take(limit as usize) {
            let Some((id, mut row)) = candidates.get(r.index).cloned() else {
                continue;
            };
            row.score = r.score;
            out.push((id, row));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires model download (~23 MB)
    fn test_embed_produces_384_dim() {
        let mut embedder = Embedder::new().expect("failed to initialize embedder");
        let embedding = embedder
            .embed_single("hello world")
            .expect("failed to embed");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    #[ignore] // requires model download (~23 MB)
    fn test_embed_batch() {
        let mut embedder = Embedder::new().expect("failed to initialize embedder");
        let embeddings = embedder
            .embed_texts(&["hello", "world"])
            .expect("failed to embed batch");
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[test]
    fn test_model_choice_values() {
        assert_eq!(ModelChoice::BgeSmallEnV15.model_id(), "bge-small-en-v1.5");
        assert_eq!(ModelChoice::BgeSmallEnV15.dimension(), 384);
        assert_eq!(ModelChoice::BgeM3.model_id(), "bge-m3");
        assert_eq!(ModelChoice::BgeM3.dimension(), 1024);
        assert_eq!(ModelChoice::default(), ModelChoice::BgeSmallEnV15);
    }

    #[test]
    fn test_openai_compatible_model_id() {
        assert_eq!(
            OpenAiCompatibleConfig::model_id("nomic-embed-text"),
            "openai:nomic-embed-text"
        );
        let settings = EmbeddingSettings::openai_compatible(OpenAiCompatibleConfig {
            endpoint: "http://127.0.0.1:11434/v1/embeddings".into(),
            model: "nomic-embed-text".into(),
            dimension: 768,
            timeout_seconds: 120,
            batch_size: 32,
            api_key: None,
        })
        .unwrap();
        assert_eq!(settings.model_id(), "openai:nomic-embed-text");
        assert_eq!(settings.dimension(), 768);
        assert!(settings.is_openai_compatible());
    }

    #[test]
    fn test_openai_compatible_rejects_bad_endpoint() {
        let err = EmbeddingSettings::openai_compatible(OpenAiCompatibleConfig {
            endpoint: "not a url".into(),
            model: "nomic-embed-text".into(),
            dimension: 768,
            timeout_seconds: 60,
            batch_size: 32,
            api_key: None,
        })
        .unwrap_err();
        assert!(err.to_string().contains("endpoint"));
    }

    #[test]
    fn test_openai_http_embed_happy_path_and_dim_mismatch() {
        let server = httpmock::MockServer::start();
        let ok = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]}
                ]
            }));
        });

        let mut embedder = Embedder::with_settings(
            EmbeddingSettings::openai_compatible(OpenAiCompatibleConfig {
                endpoint: server.url("/v1/embeddings"),
                model: "mock-model".into(),
                dimension: 3,
                timeout_seconds: 5,
                batch_size: 32,
                api_key: None,
            })
            .unwrap(),
        )
        .unwrap();
        let vectors = embedder.embed_texts(&["a", "b"]).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(vectors[1], vec![0.4, 0.5, 0.6]);
        ok.assert();

        let bad = httpmock::MockServer::start();
        bad.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/v1/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}]
            }));
        });
        let mut bad_embedder = Embedder::with_settings(
            EmbeddingSettings::openai_compatible(OpenAiCompatibleConfig {
                endpoint: bad.url("/v1/embeddings"),
                model: "mock-model".into(),
                dimension: 3,
                timeout_seconds: 5,
                batch_size: 32,
                api_key: None,
            })
            .unwrap(),
        )
        .unwrap();
        let err = bad_embedder.embed_single("x").unwrap_err();
        assert!(
            err.to_string().contains("dimension mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn test_model_choice_batch_size_is_smaller_for_large_model() {
        // 大きなモデルは activation memory が多いので batch を絞る。
        // OOM 防止のための invariant を固定化 (将来値を変えるときはここも更新)。
        assert!(
            ModelChoice::BgeM3.batch_size() < ModelChoice::BgeSmallEnV15.batch_size(),
            "BGE-M3 batch must be smaller than BGE-small-en-v1.5 batch"
        );
        assert!(ModelChoice::BgeM3.batch_size() > 0);
    }

    #[test]
    fn test_reranker_choice_values() {
        assert!(!RerankerChoice::None.is_enabled());
        assert!(RerankerChoice::BgeV2M3.is_enabled());
        assert!(RerankerChoice::JinaV2Multilingual.is_enabled());
        assert!(RerankerChoice::BgeBase.is_enabled());
        assert_eq!(RerankerChoice::default(), RerankerChoice::None);
        assert_eq!(RerankerChoice::BgeV2M3.model_id(), "bge-reranker-v2-m3");
        assert_eq!(RerankerChoice::BgeV2M3.approx_download_mb(), 2300);
    }

    #[test]
    fn test_reranker_value_enum_tag_matches_bench_arg() {
        // F-60 PR-1 codex P1 regression: benches/search_latency.rs hard-codes
        // `--reranker bge-v2-m3` as the heavy-bench subprocess argument. If
        // the `#[value(name = "...")]` tag on RerankerChoice::BgeV2M3 ever
        // diverges from this literal, the bench would fail at clap parse time.
        // The HuggingFace model id `bge-reranker-v2-m3` must NOT be a valid
        // CLI value (it lives behind `RerankerChoice::model_id()`).
        use clap::ValueEnum;
        assert!(
            RerankerChoice::from_str("bge-v2-m3", false).is_ok(),
            "CLI must accept the bench-hardcoded reranker tag 'bge-v2-m3'"
        );
        assert!(
            RerankerChoice::from_str("bge-reranker-v2-m3", false).is_err(),
            "the HuggingFace model id must not be a valid CLI value"
        );
        assert!(RerankerChoice::from_str("bge-base", false).is_ok());
        assert!(RerankerChoice::from_str("jina-v2-ml", false).is_ok());
        assert!(RerankerChoice::from_str("none", false).is_ok());
    }

    #[test]
    fn test_reranker_none_returns_none() {
        // DL を伴わない安全なテスト
        let r = Reranker::try_new(RerankerChoice::None).unwrap();
        assert!(r.is_none());
    }

    #[test]
    #[ignore] // requires BGE-reranker-v2-m3 download (~2.3 GB)
    fn test_bge_reranker_v2_m3_reorders_ja() {
        let mut r = Reranker::try_new(RerankerChoice::BgeV2M3)
            .expect("failed to load BGE-reranker-v2-m3")
            .expect("reranker should be Some");
        // SearchResult は db::SearchResult を使う
        use crate::db::SearchResult;
        let mk = |content: &str| SearchResult {
            score: 0.0,
            content: content.to_string(),
            heading: None,
            document_id: 0,
            path: "x.md".to_string(),
            title: None,
            topic: None,
            date: None,
            tags: Vec::new(),
        };
        let candidates = vec![
            (1i64, mk("天気予報の話題です")),
            (
                2,
                mk("E0382 は所有権が移動した後の値を使ったときに出るエラーです"),
            ),
            (3, mk("映画のレビューについて")),
        ];
        let out = r
            .rerank_candidates("Rust の E0382 エラーの意味は？", candidates, 3)
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(
            out[0].content.contains("E0382"),
            "top should be E0382 content, got: {}",
            out[0].content
        );
    }

    #[test]
    #[ignore] // requires BGE-M3 download (~2.3 GB)
    fn test_bge_m3_produces_1024_dim() {
        let mut embedder = Embedder::with_model(ModelChoice::BgeM3).expect("failed to load BGE-M3");
        let emb = embedder
            .embed_single("こんにちは、世界")
            .expect("failed to embed");
        assert_eq!(emb.len(), 1024);
    }
}
