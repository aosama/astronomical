//! Logits-level causality probe for chunked K2 Horizon MoVA prefill.
//!
//! Chunked prefill is faithful only when every multi-token chunk masks its own
//! future. The engine-level journey compares sampled continuations, which
//! multinomial sampling can hide; this probe compares the underlying
//! next-token distributions against a split that is clean by construction, so
//! the noise floor and any chunking divergence are measured separately.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::{ChatMessage, ChatToolChoice};
use astronomical_runtime_integration::{
    MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMemoryLimits, MlxRuntime,
    maximum_recommended_gpu_working_set_size_bytes,
};

use crate::PerformanceAttribution;

use super::K2HorizonMoVAKvState;
use super::K2HorizonMoVAWeights;
use super::model::K2HorizonMoVAModel;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const E2E_TEST_MODEL_NAMES_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../registry/e2e_test_model_names.json"
));
const K2_HORIZON_MOVA_ROLE: &str = "k2_horizon_mova";
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
));
const PROMPT_TOKEN_COUNT: usize = 200;
const KV_STATE_GROWTH_TOKENS: u32 = 256;
/// The probe deadline. The synchronous MLX work cannot be interrupted, so the
/// timeout fails the journey when the blocking task overruns and the bounded
/// runner's process-tree kill remains the hard backstop.
const PROBE_DEADLINE: Duration = Duration::from_secs(115);

#[tokio::test]
#[ignore = "requires model_directories to discover a stacked affine K2 Horizon MoVA artifact"]
async fn should_prefill_chunking_preserve_single_chunk_logits() {
    let probe_journey = tokio::task::spawn_blocking(probe_body);
    tokio::time::timeout(PROBE_DEADLINE, probe_journey)
        .await
        .expect("the causality probe should finish within 115 seconds")
        .expect("the causality probe blocking task should join");
}

fn probe_body() {
    let runtime = probe_runtime();
    let model = load_probe_model(runtime);
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&model);
    eprintln!(
        "[k2-prefill-causality-probe] prompt_tokens={}",
        prompt_token_ids.len()
    );

    // A single forward over the whole prompt runs at a zero cache offset where
    // the causal kernel aligns its mask diagonal, so it is the causally
    // correct reference by construction.
    let reference_logits = prefill_logits(&model, &prompt_token_ids, &[PROMPT_TOKEN_COUNT]);
    // A 199-plus-1 split is also clean by construction (the one-token tail is
    // causally equivalent unmasked), so its divergence from the reference is
    // the pure kernel-shape noise floor of this comparison.
    let noise_baseline_logits = prefill_logits(&model, &prompt_token_ids, &[199, 1]);
    // Chunk boundaries inside the prompt force multi-token chunks to run at a
    // nonzero cache offset, which is exactly where unmasked attention would
    // leak each chunk's own future tokens into its past.
    let chunked_logits = prefill_logits(&model, &prompt_token_ids, &[64, 64, 64, 8]);

    // The comparison runs in float32 so the divergence statistics see the
    // bfloat16 logit values exactly instead of a second rounding.
    let reference = float32_logits(&model.runtime, &reference_logits);
    let noise_baseline = float32_logits(&model.runtime, &noise_baseline_logits);
    let chunked = float32_logits(&model.runtime, &chunked_logits);
    let noise_baseline_divergence = divergence(&reference, &noise_baseline);
    let chunked_divergence = divergence(&reference, &chunked);
    eprintln!(
        "[k2-prefill-causality-probe] noise_baseline divergence={noise_baseline_divergence:?}"
    );
    eprintln!("[k2-prefill-causality-probe] chunked divergence={chunked_divergence:?}");

    let (noise_baseline_maximum, _, noise_baseline_exceedance_count_two) =
        noise_baseline_divergence;
    let (chunked_maximum, _, chunked_exceedance_count_two) = chunked_divergence;
    // The stated acceptance bound: no logit may move by more than 2.0. The
    // clean-by-construction baseline enforces the same bound on the noise
    // floor itself, so a noisy environment fails loudly instead of widening.
    assert_eq!(
        noise_baseline_exceedance_count_two, 0,
        "the noise-floor split must not move any logit by more than 2.0"
    );
    assert_eq!(
        chunked_exceedance_count_two, 0,
        "chunked prefill must not move any logit by more than 2.0 from the \
         single-chunk reference: chunked_exceedance_count_two={chunked_exceedance_count_two}"
    );
    assert!(
        chunked_maximum <= noise_baseline_maximum * 4.0 + 0.5,
        "chunked prefill diverged from the single-chunk reference beyond the \
         kernel-shape noise floor: chunked_max_abs_diff={chunked_maximum} \
         noise_floor_max_abs_diff={noise_baseline_maximum}"
    );
}

fn float32_logits(runtime: &MlxRuntime, logits: &MlxArray) -> Vec<f32> {
    runtime
        .astype(logits, MlxDtype::Float32)
        .and_then(|float32_logits| float32_logits.to_vec_f32())
        .expect("probe logits should read as float32 values")
}

fn divergence(reference: &[f32], variant: &[f32]) -> (f32, usize, usize) {
    let mut maximum_absolute_difference = 0.0_f32;
    let mut exceedance_count_half = 0_usize;
    let mut exceedance_count_two = 0_usize;
    for (reference_logit, variant_logit) in reference.iter().zip(variant.iter()) {
        let absolute_difference = (reference_logit - variant_logit).abs();
        if absolute_difference > maximum_absolute_difference {
            maximum_absolute_difference = absolute_difference;
        }
        if absolute_difference > 0.5 {
            exceedance_count_half += 1;
        }
        if absolute_difference > 2.0 {
            exceedance_count_two += 1;
        }
    }
    (
        maximum_absolute_difference,
        exceedance_count_half,
        exceedance_count_two,
    )
}

fn prefill_logits(
    model: &K2HorizonMoVAModel,
    prompt_token_ids: &[u32],
    chunk_token_counts: &[usize],
) -> MlxArray {
    let mut caches = (0..model.config.num_hidden_layers())
        .map(|_| {
            K2HorizonMoVAKvState::build(false, KV_STATE_GROWTH_TOKENS)
                .expect("probe KV state should build")
        })
        .collect::<Vec<_>>();
    let mut performance_attribution = PerformanceAttribution::disabled();
    let mut chunk_start = 0_usize;
    let mut last_hidden_states = None;
    for chunk_token_count in chunk_token_counts {
        let chunk_end = chunk_start + chunk_token_count;
        last_hidden_states = Some(
            model
                .forward(
                    &prompt_token_ids[chunk_start..chunk_end],
                    &mut caches,
                    &mut performance_attribution,
                    false,
                )
                .expect("probe prefill chunk should forward"),
        );
        chunk_start = chunk_end;
    }
    model
        .logits_for_last_token(&last_hidden_states.expect("probe prefill should run"))
        .expect("probe logits should project")
}

fn load_probe_model(runtime: MlxRuntime) -> K2HorizonMoVAModel {
    let model_directory = configured_k2_horizon_mova_model_directory();
    let mut performance_attribution = PerformanceAttribution::disabled();
    let validated_artifact = crate::K2HorizonMoVAArtifactValidator::new()
        .validate(&model_directory)
        .expect("the installed K2 artifact should validate");
    let weights =
        K2HorizonMoVAWeights::load(&runtime, &validated_artifact, &mut performance_attribution)
            .expect("K2 weights should load for the causality probe");
    K2HorizonMoVAModel {
        runtime,
        config: validated_artifact.config().clone(),
        weights,
        compiled_swiglu: MlxCompiledSwiGlu::new().expect("probe SwiGLU should compile"),
        compiled_elementwise_graphs: MlxCompiledElementwiseGraphs::new()
            .expect("probe elementwise graphs should compile"),
        sorted_expert_reduction_kernel: None,
        fused_expert_decode_kernels: None,
    }
}

fn configured_k2_horizon_mova_model_directory() -> PathBuf {
    let role_to_model_id =
        serde_json::from_str::<BTreeMap<String, String>>(E2E_TEST_MODEL_NAMES_JSON)
            .expect("registry/e2e_test_model_names.json should parse");
    let model_id = role_to_model_id
        .get(K2_HORIZON_MOVA_ROLE)
        .expect("registry/e2e_test_model_names.json must declare the k2_horizon_mova role");
    let astronomical_config = AstronomicalConfig::load_from_development_location()
        .expect("the standard Astronomical configuration should load for the causality probe");
    astronomical_config
        .find_configured_model_directory_by_id(model_id)
        .expect("the causality probe model discovery should not fail")
        .expect("the causality probe model should be discoverable")
}

fn probe_runtime() -> MlxRuntime {
    let memory_limit_bytes = maximum_recommended_gpu_working_set_size_bytes()
        .expect("MLX should expose the default GPU wired-memory working set");
    MlxRuntime::initialize(
        MlxMemoryLimits::new(memory_limit_bytes, memory_limit_bytes)
            .expect("probe memory limits should be valid"),
    )
    .expect("probe MLX runtime")
}

fn romeo_and_juliet_prompt_token_ids(model: &K2HorizonMoVAModel) -> Vec<u32> {
    let tokenizer = crate::K2HorizonMoVATokenizer::from_json_bytes(
        &std::fs::read(configured_k2_horizon_mova_model_directory().join("tokenizer.json"))
            .expect("the installed K2 tokenizer should be readable"),
        &model.config,
    )
    .expect("the K2 tokenizer should load for the causality probe");
    let rendered_prompt = crate::K2HorizonMoVAPromptRenderer::new().render(
        &[ChatMessage::User {
            content: format!(
                "Use the supplied Romeo and Juliet source as the only source.\n\n{ROMEO_AND_JULIET_SOURCE}"
            ),
            images: Vec::new(),
        }],
        &[],
        &ChatToolChoice::None,
    );
    let prompt_token_ids = tokenizer
        .encode_prompt(&rendered_prompt)
        .expect("the Romeo and Juliet prompt should encode");
    assert!(
        prompt_token_ids.len() >= PROMPT_TOKEN_COUNT,
        "the Romeo and Juliet fixture should reach {PROMPT_TOKEN_COUNT} K2 prompt tokens"
    );
    prompt_token_ids[..PROMPT_TOKEN_COUNT].to_vec()
}
