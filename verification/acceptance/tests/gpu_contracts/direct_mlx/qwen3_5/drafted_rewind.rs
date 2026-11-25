//! A drafted block rewound to its kept rows must leave the target exactly
//! where forwarding only those rows would.
//!
//! On the small dense Qwen3.5 (the same DeltaNet and attention mix as the
//! 27B): for every kept count n from 1 to 7, verify 8 rows and rewind to n,
//! then compare the next token's logits with a request that forwarded just
//! the n rows. A control that skips the rewind must differ, so the test
//! cannot pass by comparing two states that never moved.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::{
    PerformanceAttribution, Qwen3_5ArtifactValidator, Qwen3_5ResidentModel, Qwen3_5Tokenizer,
    RequestDecoderStateStack, TextModelChunkingConfiguration, VerificationLogitsMode,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use crate::common::{configured_installed_model_directory_by_id, small_dense_model_id};

pub(super) const PROMPT: &str =
    "Two households, both alike in dignity, In fair Verona, where we lay our scene,";
pub(super) const BLOCK: [u32; 8] = [279, 3_822, 315, 264, 1_550, 11, 323, 279];
/// Both sides run the same layers over the same tokens; a few bf16 units of
/// difference can come only from MLX choosing a different matmul for a
/// different row count. A missed rewind moves logits by whole units.
pub(super) const MAXIMUM_LOGIT_DIFFERENCE: f32 = 0.25;

/// The small dense model loaded on one engine, with the prompt its tests
/// prefill. The resident model unless a contract names the other.
pub(super) struct SmallTarget<Model = Qwen3_5ResidentModel> {
    pub(super) model: Model,
    pub(super) prompt_token_ids: Vec<u32>,
}

/// What loading the small dense target on either engine starts from.
pub(super) struct SmallTargetSources {
    pub(super) runtime: MlxRuntime,
    pub(super) artifact: model_serving::ValidatedQwen3_5Artifact,
    pub(super) model_directory: std::path::PathBuf,
    pub(super) prompt_token_ids: Vec<u32>,
}

pub(super) fn small_target_sources() -> SmallTargetSources {
    target_sources(
        configured_installed_model_directory_by_id(small_dense_model_id()),
        6,
    )
}

/// The sources of any installed Qwen3.5 target, under `memory_limit_gigabytes`.
pub(super) fn target_sources(
    model_directory: std::path::PathBuf,
    memory_limit_gigabytes: usize,
) -> SmallTargetSources {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            memory_limit_gigabytes * 1024 * 1024 * 1024,
            512 * 1024 * 1024,
        )
        .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 4096)
        .expect("the artifact validates");
    let tokenizer =
        Qwen3_5Tokenizer::from_validated_artifact(&artifact).expect("the tokenizer builds");
    let prompt_token_ids = tokenizer.encode_prompt(PROMPT).expect("the prompt encodes");
    SmallTargetSources {
        runtime,
        artifact,
        model_directory,
        prompt_token_ids,
    }
}

pub(super) fn load_small_target() -> SmallTarget {
    let SmallTargetSources {
        runtime,
        artifact,
        model_directory,
        prompt_token_ids,
    } = small_target_sources();
    let chunking = TextModelChunkingConfiguration::new(256, 0, 0).expect("the chunking is valid");
    let mut model = Qwen3_5ResidentModel::load_with_performance_attribution(
        runtime,
        artifact,
        &model_directory,
        true,
        chunking,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the model loads");
    model
        .materialize_complete_weights_matching_production()
        .expect("the weights materialize");
    SmallTarget {
        model,
        prompt_token_ids,
    }
}

pub(super) fn prefilled_state(target: &SmallTarget) -> RequestDecoderStateStack {
    let mut state = RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
        &target.model.astro_decoder_state_cache_layout(),
        256,
    )
    .expect("the decoder state allocates");
    forward(target, &target.prompt_token_ids, 0, &mut state);
    state
}

pub(super) fn forward(
    target: &SmallTarget,
    token_ids: &[u32],
    first_position: u32,
    state: &mut RequestDecoderStateStack,
) -> MlxArray {
    target
        .model
        .build_speculative_verification_forward(
            token_ids,
            first_position,
            state,
            &[],
            VerificationLogitsMode::LastRowOnly,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the forward runs")
        .per_position_logits
        .expect("the last row has logits")
}

pub(super) fn largest_difference(target: &SmallTarget, left: &MlxArray, right: &MlxArray) -> f32 {
    let runtime = target.model.runtime();
    let left_values = runtime
        .array_to_vec_f32(&runtime.astype(left, MlxDtype::Float32).expect("convert"))
        .expect("read");
    let right_values = runtime
        .array_to_vec_f32(&runtime.astype(right, MlxDtype::Float32).expect("convert"))
        .expect("read");
    left_values
        .iter()
        .zip(&right_values)
        .fold(0.0_f32, |largest, (left_value, right_value)| {
            largest.max((left_value - right_value).abs())
        })
}

/// Verify the 8-row block, rewind to `kept_row_count`, and return the next
/// token's logits; with `should_rewind` false, the control skips the rewind.
fn next_logits_after_verify(
    target: &SmallTarget,
    kept_row_count: usize,
    should_rewind: bool,
) -> MlxArray {
    let mut state = prefilled_state(target);
    let first_position = target.prompt_token_ids.len() as u32;
    let mut verification = target
        .model
        .verify_drafted_rows(
            &BLOCK,
            first_position,
            &mut state,
            &[],
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the block verifies");
    let kept_token_offset = first_position as i32 + kept_row_count as i32;
    if should_rewind {
        state
            .rewind_to_kept_row_boundary(
                &mut verification.row_boundaries[kept_row_count - 1].recurrent_snapshot_tensors,
                kept_token_offset,
            )
            .expect("the state rewinds");
    }
    forward(
        target,
        &[BLOCK[kept_row_count]],
        kept_token_offset as u32,
        &mut state,
    )
}

fn next_logits_after_forwarding(target: &SmallTarget, kept_row_count: usize) -> MlxArray {
    let mut state = prefilled_state(target);
    let first_position = target.prompt_token_ids.len() as u32;
    forward(target, &BLOCK[..kept_row_count], first_position, &mut state);
    forward(
        target,
        &[BLOCK[kept_row_count]],
        first_position + kept_row_count as u32,
        &mut state,
    )
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_leave_the_same_state_as_forwarding_only_the_kept_rows() {
    let target = load_small_target();
    for kept_row_count in 1..BLOCK.len() {
        let rewound = next_logits_after_verify(&target, kept_row_count, true);
        let forwarded = next_logits_after_forwarding(&target, kept_row_count);
        let difference = largest_difference(&target, &rewound, &forwarded);
        eprintln!(
            "[drafted-rewind] kept {kept_row_count} of 8: largest logit difference {difference:.4}"
        );
        assert!(
            difference <= MAXIMUM_LOGIT_DIFFERENCE,
            "kept {kept_row_count}: logits differ by {difference}"
        );
    }
}

#[test]
#[ignore = "loads the small dense Qwen3.5 on the GPU"]
#[serial]
fn should_differ_from_forwarding_the_kept_rows_when_the_rewind_is_skipped() {
    let target = load_small_target();
    let unrewound = next_logits_after_verify(&target, 3, false);
    let forwarded = next_logits_after_forwarding(&target, 3);
    let difference = largest_difference(&target, &unrewound, &forwarded);
    eprintln!("[drafted-rewind] control without rewind: largest logit difference {difference:.4}");
    assert!(
        difference > MAXIMUM_LOGIT_DIFFERENCE * 4.0,
        "the control must expose a missing rewind"
    );
}
