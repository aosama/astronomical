//! Pins the embedding model's vectors across changes to the ModernBERT trunk.
//!
//! The decision-model work (issue #2032) reuses the ModernBERT trunk that
//! serves embeddings, so the trunk was restructured to return hidden states
//! for batches. This journey proves that restructuring left embeddings alone:
//! it embeds three Romeo and Juliet excerpts with the real installed model and
//! compares every vector component against vectors recorded from the engine
//! as it was before the restructuring.
//!
//! The recorded vectors in `tests/fixtures/modernbert_embedding_reference_vectors.json`
//! come from the embedding engine at commit `7106ed92`, the last one before
//! the trunk was restructured, run on the same three excerpts.
//!
//! Run it with `cargo test-modernbert-embedding-vectors --configuration PATH`,
//! where the configuration names the embedding model's directory.
//!
//! The three lengths are chosen to cross the trunk's branches: a short input
//! that never needs a local-attention mask, one just past the local window,
//! and a long one.

use astronomical_ipc_protocol::{EmbeddingEncodingFormat, EmbeddingsCommand, RequestId};
use astronomical_model_serving::{EmbeddingEngine, ModernBertEmbeddingEngine};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const EMBEDDING_MODEL_ID: &str = "nomicai-modernbert-embed-base-8bit";
/// Word counts of the three excerpts: below the local-attention window, just
/// past it, and well past it.
const EXCERPT_WORD_COUNTS: [usize; 3] = [8, 120, 600];
/// The embedding model is about 160 MB; this ceiling leaves ample forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 4 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// The largest component difference accepted against the recorded vectors.
///
/// The vectors are unit length, so components are below 1. On the machine
/// that recorded them the restructured trunk reproduces them exactly; the
/// tolerance only allows for a different graphics processor or MLX build
/// rounding the same arithmetic differently.
const MAXIMUM_COMPONENT_DIFFERENCE: f32 = 1e-4;
const RECORDED_VECTORS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/modernbert_embedding_reference_vectors.json"
));

fn embed_the_excerpts() -> Vec<Vec<f32>> {
    let model_directory = TestRunContext::current().required_model_directory();
    let attribution_log_directory =
        tempfile::tempdir().expect("a temporary directory for the unused attribution log");
    let mut embedding_engine = ModernBertEmbeddingEngine::new(
        EMBEDDING_MODEL_ID,
        &model_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
        false,
        attribution_log_directory.path().join("attribution.jsonl"),
    )
    .expect("the installed embedding artifact should validate");
    embedding_engine
        .load()
        .expect("the embedding model should load");
    let mut excerpts = Vec::with_capacity(EXCERPT_WORD_COUNTS.len());
    for excerpt_word_count in EXCERPT_WORD_COUNTS {
        excerpts.push(romeo_and_juliet::excerpt(excerpt_word_count));
    }
    let embeddings_output = embedding_engine
        .embed(&EmbeddingsCommand {
            request_id: RequestId::new(1),
            model: EMBEDDING_MODEL_ID.to_owned(),
            inputs: excerpts,
            encoding_format: EmbeddingEncodingFormat::Float,
            dimensions: None,
        })
        .expect("the excerpts should embed");
    eprintln!(
        "[modernbert-vectors] input_token_counts={:?}",
        embeddings_output.input_token_counts
    );
    embeddings_output.embeddings
}

#[ignore = "requires the nomicai-modernbert-embed-base-8bit model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_reproduce_the_embedding_vectors_recorded_before_the_trunk_was_restructured() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let recorded_vectors: Vec<Vec<f32>> = serde_json::from_str(RECORDED_VECTORS_JSON)
        .expect("the recorded vectors fixture should parse");
    let embedded_vectors = embed_the_excerpts();
    assert_eq!(embedded_vectors.len(), recorded_vectors.len());
    let mut largest_component_difference = 0.0_f32;
    for (embedded_vector, recorded_vector) in embedded_vectors.iter().zip(&recorded_vectors) {
        assert_eq!(embedded_vector.len(), recorded_vector.len());
        for (embedded_component, recorded_component) in embedded_vector.iter().zip(recorded_vector)
        {
            largest_component_difference =
                largest_component_difference.max((embedded_component - recorded_component).abs());
        }
    }
    eprintln!("[modernbert-vectors] largest_component_difference={largest_component_difference:e}");
    assert!(
        largest_component_difference <= MAXIMUM_COMPONENT_DIFFERENCE,
        "the largest component difference {largest_component_difference:e} exceeds {MAXIMUM_COMPONENT_DIFFERENCE:e}"
    );
}
