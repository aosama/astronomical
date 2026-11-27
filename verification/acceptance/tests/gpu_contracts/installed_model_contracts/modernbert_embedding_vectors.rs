//! Pins the embedding model's vectors across changes to the ModernBERT trunk.
//!
//! The decision-model work (issue #2032) reuses the ModernBERT trunk that
//! serves embeddings, so the trunk was restructured to return hidden states
//! for batches. This journey proves that restructuring left embeddings alone:
//! it embeds three Romeo and Juliet excerpts with the real installed model and
//! compares every vector against vectors recorded from the engine as it was
//! before the restructuring, component by component and by cosine.
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

use ipc_protocol::{EmbeddingEncodingFormat, EmbeddingsCommand, RequestId};
use model_serving::{EmbeddingEngine, FamilyEngineSettings, ModernBertEmbeddingEngine};
use test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const EMBEDDING_MODEL_ID: &str = "nomicai-modernbert-embed-base-8bit";
/// Word counts of the three excerpts: below the local-attention window, just
/// past it, and well past it.
const EXCERPT_WORD_COUNTS: [usize; 3] = [8, 120, 600];
/// The embedding model is about 160 MB; this ceiling leaves ample forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 4 * 1024 * 1024 * 1024;
const TEST_MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// The largest component difference accepted against the recorded vectors.
///
/// The vectors are unit length, so components are below 1. They were
/// recorded on Metal, where the restructured trunk reproduces them exactly,
/// so Metal keeps a tolerance that only allows for another Apple GPU
/// rounding the same arithmetic differently.
///
/// MLX's CUDA and CPU backends sum the same float32 products in other
/// orders, and through 22 layers of 8-bit projections that moves components
/// by up to about 1e-3. Measured on the three excerpts (issue #2699): CUDA
/// against CPU 1.1e-4 to 1.7e-4, CUDA against the Metal recording 1.6e-4 to
/// 9.5e-4, CPU against it 1.6e-4 to 8.4e-4, and every pair's cosine at least
/// 0.999995. No backend is the odd one out, so elsewhere the tolerance is
/// sized to that spread, and the cosine below does the pinning.
#[cfg(target_os = "macos")]
const MAXIMUM_COMPONENT_DIFFERENCE: f32 = 1e-4;
#[cfg(not(target_os = "macos"))]
const MAXIMUM_COMPONENT_DIFFERENCE: f32 = 2e-3;
/// The smallest cosine accepted between an embedded vector and its
/// recording: ten times closer to 1 than the backends' measured spread
/// allows for, and far above what a changed trunk gives (a wrong mask or a
/// skipped layer turns the vector, not its last digits).
const MINIMUM_COSINE_WITH_RECORDING: f64 = 0.99999;
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
        &FamilyEngineSettings {
            effective_mlx_memory_ceiling_bytes: TEST_MLX_MEMORY_CEILING_BYTES,
            mlx_buffer_cache_limit_bytes: TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
            floor_raise_limit_bytes: 0,
            performance_attribution_enabled: false,
            performance_attribution_log_path: attribution_log_directory
                .path()
                .join("attribution.jsonl"),
        },
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
    let mut smallest_cosine = 1.0_f64;
    for (embedded_vector, recorded_vector) in embedded_vectors.iter().zip(&recorded_vectors) {
        assert_eq!(embedded_vector.len(), recorded_vector.len());
        for (embedded_component, recorded_component) in embedded_vector.iter().zip(recorded_vector)
        {
            largest_component_difference =
                largest_component_difference.max((embedded_component - recorded_component).abs());
        }
        smallest_cosine = smallest_cosine.min(cosine(embedded_vector, recorded_vector));
    }
    eprintln!(
        "[modernbert-vectors] largest_component_difference={largest_component_difference:e} smallest_cosine={smallest_cosine:.9}"
    );
    assert!(
        largest_component_difference <= MAXIMUM_COMPONENT_DIFFERENCE,
        "the largest component difference {largest_component_difference:e} exceeds {MAXIMUM_COMPONENT_DIFFERENCE:e}"
    );
    assert!(
        smallest_cosine >= MINIMUM_COSINE_WITH_RECORDING,
        "a vector's cosine with its recording is {smallest_cosine:.9}, below {MINIMUM_COSINE_WITH_RECORDING}"
    );
}

fn cosine(left_vector: &[f32], right_vector: &[f32]) -> f64 {
    let (mut dot_product, mut left_squared, mut right_squared) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (left_component, right_component) in left_vector.iter().zip(right_vector) {
        dot_product += f64::from(*left_component) * f64::from(*right_component);
        left_squared += f64::from(*left_component) * f64::from(*left_component);
        right_squared += f64::from(*right_component) * f64::from(*right_component);
    }
    dot_product / (left_squared.sqrt() * right_squared.sqrt())
}
