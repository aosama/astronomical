//! GPU unit tests on the sealed fixture pack: the family model loads the
//! tiny fixture and runs the straight forward pass end to end. Real-artifact
//! journeys (upstream parity, text-in/text-out) live in the acceptance crate
//! under `verification/acceptance/tests/gpu_contracts/prism_hadamard.rs`.

use std::collections::HashMap;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::forward::StraightGenerationState;
use super::{PrismHadamardModelError, PrismHadamardTextModel, validate_greedy_generation_request};
use crate::prism_hadamard_qwen35::artifacts::FoldedPackValidator;
use crate::prism_hadamard_qwen35::artifacts::tests::pack_fixture::{
    FIXTURE_VOCAB_SIZE, remove_fixture_directory, write_fixture_pack,
};

const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../verification/acceptance/tests/fixtures/model_metrics_10000_romeo_and_juliet_words.txt"
);

/// The fixture weights are a few megabytes, so a modest wired limit keeps the
/// smoke tests far away from platform admission behavior they do not exercise.
fn smoke_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the smoke test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

#[test]
#[serial]
fn should_project_only_the_final_prompt_position_to_vocabulary_logits() {
    let fixture_directory = write_fixture_pack(|_| {});
    let pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let model = PrismHadamardTextModel::new(smoke_test_runtime(), pack)
        .expect("the fixture pack should load into the runtime");

    let token_indices = romeo_and_juliet_fixture_token_ids(5);
    let tokens = model
        .runtime
        .array_from_i32(&token_indices, &[1, token_indices.len() as i32])
        .expect("the smoke test tokens should build");
    let mut state = StraightGenerationState::empty();
    let final_hidden_states = model
        .forward_hidden_states(&tokens, &mut state)
        .expect("the decoder should return the final prompt hidden row");
    assert_eq!(
        final_hidden_states.shape(),
        vec![1, 1, 512],
        "only the final prompt row should reach the vocabulary projection"
    );
    let logits = model
        .logits_for_last_hidden_state(&final_hidden_states)
        .expect("the final hidden row should project to vocabulary logits");
    model
        .runtime
        .evaluate_arrays(&[&logits])
        .expect("the logits should evaluate");

    assert_eq!(
        logits.shape(),
        vec![1, 1, FIXTURE_VOCAB_SIZE as i32],
        "the vocabulary projection should contain only the final prompt position"
    );

    remove_fixture_directory(&fixture_directory);
}

#[test]
#[serial]
fn should_continue_generation_state_across_decoding_steps() {
    let fixture_directory = write_fixture_pack(|_| {});
    let pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let model = PrismHadamardTextModel::new(smoke_test_runtime(), pack)
        .expect("the fixture pack should load into the runtime");

    let mut state = StraightGenerationState::empty();
    let prompt_token_ids = romeo_and_juliet_fixture_token_ids(3);
    let prompt_tokens = model
        .runtime
        .array_from_i32(&prompt_token_ids, &[1, prompt_token_ids.len() as i32])
        .expect("the prompt tokens should build");
    let prompt_logits = model
        .forward_pass(&prompt_tokens, &mut state)
        .expect("the prefill pass should run");
    model
        .runtime
        .evaluate_arrays(&[&prompt_logits])
        .expect("the prefill logits should evaluate");
    assert_eq!(prompt_logits.shape(), vec![1, 1, FIXTURE_VOCAB_SIZE as i32]);

    let decode_token_id = romeo_and_juliet_fixture_token_ids(4)
        .last()
        .copied()
        .expect("the source excerpt should contain a continuation word");
    let decode_tokens = model
        .runtime
        .array_from_i32(&[decode_token_id], &[1, 1])
        .expect("the decode token should build");
    let decode_logits = model
        .forward_pass(&decode_tokens, &mut state)
        .expect("the decode pass should reuse the generation state");
    model
        .runtime
        .evaluate_arrays(&[&decode_logits])
        .expect("the decode logits should evaluate");
    assert_eq!(
        decode_logits.shape(),
        vec![1, 1, FIXTURE_VOCAB_SIZE as i32],
        "the decode pass must produce exactly one logit row"
    );

    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_accept_only_single_stage_power_of_two_blocks_for_the_fused_rotation() {
    use super::signed_rotation_kernel::is_signed_rotation_eligible;
    // One radix wide up to MLX's single-stage limit is reproducible.
    assert!(is_signed_rotation_eligible(16));
    assert!(is_signed_rotation_eligible(512));
    assert!(is_signed_rotation_eligible(1024));
    assert!(is_signed_rotation_eligible(8192));
    // Narrower than one radix, non-power-of-two, or wider than one stage
    // would follow a different MLX schedule, so the composed route keeps them.
    assert!(!is_signed_rotation_eligible(8));
    assert!(!is_signed_rotation_eligible(1536));
    assert!(!is_signed_rotation_eligible(16384));
    assert!(!is_signed_rotation_eligible(0));
}

#[test]
#[serial]
fn should_rotate_bit_identically_through_the_fused_kernel_and_the_composed_route() {
    let fixture_directory = write_fixture_pack(|_| {});
    let pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the fixture pack should validate");
    let model = PrismHadamardTextModel::new(smoke_test_runtime(), pack)
        .expect("the fixture pack should load into the runtime");
    let verdict = crate::kernel_capability::worker_process_kernel_capabilities(
        &model.runtime,
        &mut crate::PerformanceAttribution::disabled(),
    )
    .verdict(crate::CustomMetalKernelFamily::PRISM_HADAMARD_SIGNED_ROTATION);
    assert_eq!(
        verdict,
        crate::CustomKernelVerdict::Supported,
        "the fused rotation probe must pass on the test GPU"
    );
    let fused_rotation_kernel = model
        .fused_rotation_kernel
        .as_ref()
        .expect("the fixture block size is eligible, so the model must retain the kernel");

    // The fixture's packed weights are zero-filled, so its embeddings cannot
    // exercise the arithmetic; the activations come from the Romeo and
    // Juliet source bytes instead, at the hidden width (one block per row)
    // and the intermediate width (two blocks per row, so the per-block sign
    // indexing is covered).
    let hidden_width = model.pack().config().text().hidden_size as i32;
    let intermediate_width = model.pack().config().text().intermediate_size as i32;
    for width in [hidden_width, intermediate_width] {
        let row_count = 7;
        let activation_values: Vec<f32> = ROMEO_AND_JULIET_SOURCE
            .bytes()
            .take((row_count * width) as usize)
            .map(|source_byte| (f32::from(source_byte) - 96.0) / 32.0)
            .collect();
        assert_eq!(activation_values.len(), (row_count * width) as usize);
        let activations = model
            .runtime
            .array_from_f32(&activation_values, &[1, row_count, width])
            .and_then(|float32_activations| {
                model.runtime.astype(
                    &float32_activations,
                    astronomical_mlx_c_rust::MlxDtype::Float16,
                )
            })
            .expect("the Romeo and Juliet activations should build");
        let fused = model
            .weights
            .rotate(
                &model.runtime,
                &model.compiled_elementwise_graphs,
                Some(fused_rotation_kernel),
                &activations,
            )
            .expect("the fused rotation should build")
            .into_activations();
        let composed = model
            .weights
            .rotate(
                &model.runtime,
                &model.compiled_elementwise_graphs,
                None,
                &activations,
            )
            .expect("the composed rotation should build")
            .into_activations();
        let fused_values = model
            .runtime
            .astype(&fused, astronomical_mlx_c_rust::MlxDtype::Float32)
            .and_then(|array| array.to_vec_f32().map_err(Into::into))
            .expect("the fused rotation should evaluate");
        let composed_values = model
            .runtime
            .astype(&composed, astronomical_mlx_c_rust::MlxDtype::Float32)
            .and_then(|array| array.to_vec_f32().map_err(Into::into))
            .expect("the composed rotation should evaluate");
        assert_eq!(fused_values.len(), composed_values.len());
        assert!(
            fused_values.iter().any(|value| *value != 0.0),
            "the rotation of text-derived activations is never all zeros"
        );
        for (value_index, (fused_value, composed_value)) in
            fused_values.iter().zip(&composed_values).enumerate()
        {
            assert_eq!(
                fused_value.to_bits(),
                composed_value.to_bits(),
                "rotated value {value_index} at width {width} differs between the fused kernel and the composed route"
            );
        }
    }

    remove_fixture_directory(&fixture_directory);
}

/// The sealed model fixture has no deployable tokenizer, so its embedding rows
/// receive stable ids allocated from the Romeo-and-Juliet source words.
fn romeo_and_juliet_fixture_token_ids(maximum_word_count: usize) -> Vec<i32> {
    let mut fixture_word_to_token_id = HashMap::new();
    ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(maximum_word_count)
        .map(|fixture_word| {
            let next_token_id = fixture_word_to_token_id.len() as u32;
            let token_id = *fixture_word_to_token_id
                .entry(fixture_word)
                .or_insert(next_token_id);
            i32::try_from(token_id).expect("the short fixture vocabulary should fit the model")
        })
        .collect()
}

#[test]
fn should_refuse_an_empty_public_greedy_prompt() {
    assert!(matches!(
        validate_greedy_generation_request(&[], 1, FIXTURE_VOCAB_SIZE, 4096),
        Err(PrismHadamardModelError::InvalidGenerationRequest { .. })
    ));
}

#[test]
fn should_refuse_a_public_greedy_prompt_token_outside_the_vocabulary() {
    assert!(matches!(
        validate_greedy_generation_request(
            &[FIXTURE_VOCAB_SIZE as i32],
            1,
            FIXTURE_VOCAB_SIZE,
            4096
        ),
        Err(PrismHadamardModelError::InvalidGenerationRequest { .. })
    ));
}

#[test]
fn should_refuse_a_public_greedy_prompt_and_output_budget_beyond_context() {
    assert!(matches!(
        validate_greedy_generation_request(&[1, 2], 3, FIXTURE_VOCAB_SIZE, 4),
        Err(PrismHadamardModelError::InvalidGenerationRequest { .. })
    ));
}
