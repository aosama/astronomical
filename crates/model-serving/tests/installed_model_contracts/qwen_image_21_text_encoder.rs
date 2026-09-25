//! Direct-MLX acceptance journeys for the native Qwen3-VL text encoder.
//!
//! These ignored journeys load the reviewed artifact's real `text_encoder/model.safetensors`
//! (~4.1 GB of 4-bit affine-quantized language-model weights; the vision tower and `lm_head`
//! are not bound for the text-only path) and encode a real prompt end to end. The token ids
//! come from the artifact's real tokenizer driven through the family's conditioning package —
//! the same path the render pipeline will take.
//!
//! The journeys prove execution-level properties: the quantized stack loads and runs, a
//! tokenized prompt produces exactly `(1, tokens, 4096)` hidden states in BF16, encoding is
//! deterministic, different prompts produce different conditioning, and out-of-vocabulary ids
//! are rejected. Numeric equality with the transformers reference is a property of the future
//! end-to-end render journey.

use std::fs::File;
use std::time::Duration;

use astronomical_model_serving::{
    QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH, QwenImage21TextEncoder, normalize_empty_prompt,
    render_t2i_prompt_template,
};
use astronomical_runtime_integration::{MlxDtype, MlxRuntime};
use tokenizers::Tokenizer;

use crate::common::qwen_image_21::{component_weights_path, shared_journey_runtime};

const ENCODER_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

/// A short prompt from the repo's mandated source text.
const PROMPT: &str = "Romeo, Romeo, wherefore art thou Romeo?";

fn artifact_tokenizer() -> Tokenizer {
    let processor_directory = crate::common::qwen_image_21::artifact_directory().join("processor");
    let tokenizer_path = processor_directory.join("tokenizer.json");
    assert!(
        tokenizer_path.exists(),
        "expected the Qwen-Image-2.1 artifact tokenizer at {}",
        tokenizer_path.display()
    );
    Tokenizer::from_file(tokenizer_path).expect("the artifact tokenizer should load")
}

/// The full token ids of a rendered text-to-image prompt, via the real tokenizer — exactly
/// what the pipeline hands the vision-language encoder before any packaging happens.
fn prompt_token_ids(prompt: &str) -> Vec<u32> {
    let tokenizer = artifact_tokenizer();
    let template = render_t2i_prompt_template(&normalize_empty_prompt(prompt));
    tokenizer
        .encode(template, false)
        .expect("the rendered template should encode")
        .get_ids()
        .to_vec()
}

fn encode_prompt(runtime: &MlxRuntime, encoder: &QwenImage21TextEncoder, prompt: &str) -> Vec<f32> {
    let hidden_states = encoder
        .encode(runtime, &prompt_token_ids(prompt))
        .expect("the encoder should encode the prompt");
    let shape = hidden_states.shape();
    let token_count = prompt_token_ids(prompt).len() as i32;
    assert_eq!(
        shape,
        vec![1, token_count, QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH as i32],
        "the encoder must emit one {}-wide hidden state per prompt token",
        QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH
    );
    runtime
        .astype(&hidden_states, MlxDtype::Float32)
        .expect("the hidden states should cast to float32")
        .to_vec_f32()
        .expect("the hidden states should materialize")
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_encode_a_real_prompt_into_deterministic_conditioning() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(ENCODER_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let encoder = QwenImage21TextEncoder::load(
            &runtime,
            File::open(component_weights_path("text_encoder"))
                .expect("the text-encoder weights should open"),
        )
        .expect("the Qwen3-VL text encoder should load from the real artifact");

        let hidden_states = encode_prompt(&runtime, &encoder, PROMPT);
        assert!(
            hidden_states.iter().all(|value| value.is_finite()),
            "every hidden-state channel must be finite"
        );
        assert!(
            hidden_states.iter().any(|value| *value != hidden_states[0]),
            "a real prompt must not encode to one constant value"
        );

        let reencoded = encode_prompt(&runtime, &encoder, PROMPT);
        assert_eq!(
            hidden_states, reencoded,
            "repeated encodings must be deterministic"
        );

        let other_prompt =
            encode_prompt(&runtime, &encoder, "Deny thy father and refuse thy name;");
        assert_ne!(
            hidden_states, other_prompt,
            "different prompts must produce different conditioning"
        );
    })
    .await
    .expect("the encoder journey should finish within its 115 s budget");
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_reject_an_out_of_vocabulary_token_id() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(ENCODER_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let encoder = QwenImage21TextEncoder::load(
            &runtime,
            File::open(component_weights_path("text_encoder"))
                .expect("the text-encoder weights should open"),
        )
        .expect("the Qwen3-VL text encoder should load from the real artifact");

        let rejection = encoder.encode(&runtime, &[u32::MAX]);
        assert!(
            rejection.is_err(),
            "an out-of-vocabulary token id must be rejected, not embedded"
        );
    })
    .await
    .expect("the rejection journey should finish within its 115 s budget");
}
