//! Direct-MLX acceptance journeys for the native Qwen-Image-2.1 denoising transformer.
//!
//! These ignored journeys load the reviewed artifact's real `transformer/model.safetensors`
//! (761 tensors: 32 blocks of 4-bit affine-quantized projections plus the shared modulation
//! network, ~4.0 GB wired) and drive the ported non-cached forward — prefill semantics with the
//! exact multi-segment block-causal attention — end to end on the GPU.
//!
//! The journeys prove execution-level properties: the quantized graph loads and runs, a
//! text-conditioned latent request produces the exact target-token geometry, outputs are
//! deterministic, and the shared modulation actually modulates (different timesteps and
//! different latents produce different predictions). The text embeddings are deterministic
//! synthetic tensors, not the real Qwen3-VL encoder output — the encoder slice lands after the
//! transformer — so numeric equality with the diffusers reference is a property of the future
//! end-to-end render journey, not this one.
//!
//! Resolves the artifact through `ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY`, and derives
//! the wired-memory limit from the artifact's own weight file size plus fixed activation
//! headroom, so the journey adapts to any artifact revision instead of any machine.

use std::fs::File;
use std::time::Duration;

use astronomical_model_serving::{
    QWEN_IMAGE_21_LATENT_CHANNEL_COUNT, QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH, QwenImage21Transformer,
    QwenImage21TransformerRequest,
};
use astronomical_runtime_integration::{MlxDtype, MlxRuntime};

use crate::common::qwen_image_21::{component_weights_path, shared_journey_runtime};

const TRANSFORMER_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

/// Target latent grid: 8×8 tokens — the smallest size every attention segment handles.
const TARGET_HEIGHT: usize = 8;
const TARGET_WIDTH: usize = 8;
const TARGET_TOKENS: usize = TARGET_HEIGHT * TARGET_WIDTH;
/// The channel and embedding widths come from the family's geometry constants rather than
/// restated numbers, so this journey checks the transformer against the same facts the
/// pipeline and the VAE use.
const LATENT_CHANNEL_COUNT: usize = QWEN_IMAGE_21_LATENT_CHANNEL_COUNT;
const TEXT_TOKEN_COUNT: usize = 41;
const TEXT_EMBEDDING_WIDTH: usize = QWEN_IMAGE_21_TEXT_EMBEDDING_WIDTH;

/// Deterministic, spatially varying packed latents `(1, 64, 64)`.
fn journey_latents(runtime: &MlxRuntime) -> astronomical_runtime_integration::MlxArray {
    let position_count = TARGET_TOKENS * LATENT_CHANNEL_COUNT;
    let latent_values: Vec<f32> = (0..position_count)
        .map(|index| ((index % 251) as f32 / 251.0 - 0.5) * 6.0)
        .collect();
    let latents = runtime
        .array_from_f32(
            &latent_values,
            &[1, TARGET_TOKENS as i32, LATENT_CHANNEL_COUNT as i32],
        )
        .expect("the journey latents should materialize");
    runtime
        .astype(&latents, MlxDtype::BFloat16)
        .expect("the journey latents should cast to the activation dtype")
}

/// Deterministic synthetic text embeddings `(1, 41, 4096)` standing in for the Qwen3-VL encoder
/// output until that slice lands.
fn journey_text_embeddings(runtime: &MlxRuntime) -> astronomical_runtime_integration::MlxArray {
    let position_count = TEXT_TOKEN_COUNT * TEXT_EMBEDDING_WIDTH;
    let embedding_values: Vec<f32> = (0..position_count)
        .map(|index| ((index % 509) as f32 / 509.0 - 0.5) * 2.0)
        .collect();
    let embeddings = runtime
        .array_from_f32(
            &embedding_values,
            &[1, TEXT_TOKEN_COUNT as i32, TEXT_EMBEDDING_WIDTH as i32],
        )
        .expect("the journey text embeddings should materialize");
    runtime
        .astype(&embeddings, MlxDtype::BFloat16)
        .expect("the journey text embeddings should cast to the activation dtype")
}

/// A text-only request: no condition images, one 8×8 target, no padding.
fn journey_request<'a>(
    latents: &'a astronomical_runtime_integration::MlxArray,
    text_embeddings: &'a astronomical_runtime_integration::MlxArray,
    timesteps: &'a [f32],
) -> QwenImage21TransformerRequest<'a> {
    QwenImage21TransformerRequest {
        packed_latents: latents,
        text_embeddings,
        vlm_image_mask: &[false; TEXT_TOKEN_COUNT],
        img_shapes: &[(TARGET_HEIGHT, TARGET_WIDTH)],
        timesteps,
    }
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_denoise_a_text_conditioned_latent_grid_deterministically() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(TRANSFORMER_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let transformer = QwenImage21Transformer::load(
            &runtime,
            File::open(component_weights_path("transformer"))
                .expect("the transformer weights should open"),
        )
        .expect("the Qwen-Image-2.1 transformer should load from the real artifact");

        let latents = journey_latents(&runtime);
        let text_embeddings = journey_text_embeddings(&runtime);

        let prediction = transformer
            .forward(
                &runtime,
                &journey_request(&latents, &text_embeddings, &[0.5]),
            )
            .expect("the transformer should denoise the journey latents");
        assert_eq!(
            prediction.shape(),
            vec![1, TARGET_TOKENS as i32, LATENT_CHANNEL_COUNT as i32],
            "the forward must return exactly the target-image token slice"
        );
        let prediction_values = runtime
            .astype(&prediction, MlxDtype::Float32)
            .expect("the prediction should cast to float32")
            .to_vec_f32()
            .expect("the prediction should materialize");
        assert_eq!(
            prediction_values.len(),
            TARGET_TOKENS * LATENT_CHANNEL_COUNT
        );
        assert!(
            prediction_values.iter().all(|value| value.is_finite()),
            "every predicted latent channel must be finite"
        );
        assert!(
            prediction_values
                .iter()
                .any(|value| *value != prediction_values[0]),
            "a varying latent request must not predict one constant value"
        );

        let reprediction = transformer
            .forward(
                &runtime,
                &journey_request(&latents, &text_embeddings, &[0.5]),
            )
            .expect("the second forward should succeed");
        let reprediction_values = runtime
            .astype(&reprediction, MlxDtype::Float32)
            .expect("the reprediction should cast to float32")
            .to_vec_f32()
            .expect("the reprediction should materialize");
        assert_eq!(
            prediction_values, reprediction_values,
            "repeated forwards must be deterministic"
        );

        // The shared modulation must actually modulate: a different timestep changes the
        // prediction even though latents and text are unchanged.
        let other_timestep_prediction = transformer
            .forward(
                &runtime,
                &journey_request(&latents, &text_embeddings, &[0.9]),
            )
            .expect("the shifted-timestep forward should succeed");
        let other_timestep_values = runtime
            .astype(&other_timestep_prediction, MlxDtype::Float32)
            .expect("the shifted-timestep prediction should cast to float32")
            .to_vec_f32()
            .expect("the shifted-timestep prediction should materialize");
        assert_ne!(
            prediction_values, other_timestep_values,
            "the timestep conditioning must influence the prediction"
        );

        // And the latents must matter too: different noise, different prediction.
        let mut shifted_latent_values: Vec<f32> = (0..TARGET_TOKENS * LATENT_CHANNEL_COUNT)
            .map(|index| ((index % 241) as f32 / 241.0 - 0.5) * 6.0)
            .collect();
        shifted_latent_values[0] += 1.25;
        let shifted_latents = runtime
            .astype(
                &runtime
                    .array_from_f32(
                        &shifted_latent_values,
                        &[1, TARGET_TOKENS as i32, LATENT_CHANNEL_COUNT as i32],
                    )
                    .expect("the shifted latents should materialize"),
                MlxDtype::BFloat16,
            )
            .expect("the shifted latents should cast");
        let shifted_prediction = transformer
            .forward(
                &runtime,
                &journey_request(&shifted_latents, &text_embeddings, &[0.5]),
            )
            .expect("the shifted-latent forward should succeed");
        let shifted_values = runtime
            .astype(&shifted_prediction, MlxDtype::Float32)
            .expect("the shifted-latent prediction should cast to float32")
            .to_vec_f32()
            .expect("the shifted-latent prediction should materialize");
        assert_ne!(
            prediction_values, shifted_values,
            "the latent input must influence the prediction"
        );
    })
    .await
    .expect("the transformer journey should finish within its 115 s budget");
}

#[ignore = "requires the Qwen-Image-2.1 artifact directory; set ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY"]
#[tokio::test]
async fn should_reject_a_latent_request_with_the_wrong_channel_count() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    tokio::time::timeout(TRANSFORMER_JOURNEY_TIMEOUT, async {
        let runtime = shared_journey_runtime();
        let transformer = QwenImage21Transformer::load(
            &runtime,
            File::open(component_weights_path("transformer"))
                .expect("the transformer weights should open"),
        )
        .expect("the Qwen-Image-2.1 transformer should load from the real artifact");

        let malformed = runtime
            .array_from_f32(&vec![0.0; TARGET_TOKENS * 4], &[1, TARGET_TOKENS as i32, 4])
            .expect("the malformed latents should materialize");
        let text_embeddings = journey_text_embeddings(&runtime);
        let rejection = transformer.forward(
            &runtime,
            &journey_request(&malformed, &text_embeddings, &[0.5]),
        );
        assert!(
            rejection.is_err(),
            "latents with four channels must be rejected, not decoded"
        );
    })
    .await
    .expect("the rejection journey should finish within its 115 s budget");
}
