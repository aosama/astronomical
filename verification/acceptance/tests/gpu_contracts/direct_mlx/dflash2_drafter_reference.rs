//! The Rust DFlash2 drafter against z-lab's own MLX reference.
//!
//! `fixtures/dflash2_reference` holds a tiny drafter with random weights run
//! through the reference `dflash/model_mlx.py` for two rounds (its generator
//! sits beside it). The second round appends context past the 8-row sliding
//! window and starts at position 12, so the window cut, absolute RoPE
//! positions and context growth are all exercised. Both run in float32, so the
//! hidden rows and logits must agree to float32 rounding and the greedy path
//! and kept candidates must agree exactly.

use std::fs::File;
use std::path::PathBuf;

use astronomical_model_serving::speculative_decoding::dflash2_drafter::{
    DFlash2DraftTarget, DFlash2DrafterConfiguration, DrafterContextStates, PathChoice,
    bind_drafter_weights, drafted_hidden_rows, select_path,
};
use astronomical_runtime_integration::{
    MlxMemoryLimits, MlxRuntime, MlxRuntimeError, MlxSafetensors,
};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

/// Float32 sums in a different order: a few units in the last place of
/// values around one, after five stacked layers.
const MAXIMUM_DIFFERENCE_OVER_LARGEST_VALUE: f32 = 1e-4;

fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash2_reference")
}

/// The fixture's target: a plain embedding table and head.
struct FixtureTarget {
    embedding_table: MlxArray,
    transposed_head: MlxArray,
}

impl DFlash2DraftTarget for FixtureTarget {
    fn embedded_rows(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let row_count = token_ids.shape()[1];
        let flat_ids = runtime.reshape(token_ids, &[row_count])?;
        let rows = runtime.take_axis(&self.embedding_table, &flat_ids, 0)?;
        let hidden_size = self.embedding_table.shape()[1];
        Ok(runtime.reshape(&rows, &[1, row_count, hidden_size])?)
    }

    fn head_logits(
        &self,
        runtime: &MlxRuntime,
        hidden_rows: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        Ok(runtime.matmul(hidden_rows, &self.transposed_head)?)
    }
}

fn values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

fn assert_close(runtime: &MlxRuntime, candidate: &MlxArray, reference: &MlxArray, label: &str) {
    assert_eq!(candidate.shape(), reference.shape(), "{label} shape");
    let candidate_values = values(runtime, candidate);
    let reference_values = values(runtime, reference);
    let mut largest_reference = 0.0_f32;
    let mut largest_difference = 0.0_f32;
    for (candidate_value, reference_value) in candidate_values.iter().zip(&reference_values) {
        largest_reference = largest_reference.max(reference_value.abs());
        largest_difference = largest_difference.max((candidate_value - reference_value).abs());
    }
    let relative_difference = largest_difference / largest_reference;
    eprintln!(
        "[dflash2-reference] {label}: largest difference {relative_difference:.2e} of the largest value"
    );
    assert!(
        relative_difference <= MAXIMUM_DIFFERENCE_OVER_LARGEST_VALUE,
        "{label}: differs by {relative_difference} of the largest value"
    );
}

/// The kept candidates per slot as sorted ids: argpartition's order within
/// the kept set is not part of the contract.
fn sorted_candidates_per_slot(runtime: &MlxRuntime, candidate_ids: &MlxArray) -> Vec<Vec<u32>> {
    let candidate_count = candidate_ids.shape()[2] as usize;
    let mut per_slot = Vec::new();
    for slot_values in values(runtime, candidate_ids).chunks(candidate_count) {
        let mut slot_ids: Vec<u32> = slot_values.iter().map(|value| *value as u32).collect();
        slot_ids.sort_unstable();
        per_slot.push(slot_ids);
    }
    per_slot
}

#[test]
#[ignore = "runs MLX on the GPU"]
#[serial]
fn should_reproduce_z_labs_reference_drafter_over_two_rounds() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let config_document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_directory().join("dflash2_reference_config.json"))
            .expect("the config reads"),
    )
    .expect("the config parses");
    let configuration = DFlash2DrafterConfiguration::from_config_document(&config_document)
        .expect("the configuration is accepted");
    let checkpoint: MlxSafetensors = runtime
        .load_safetensors(
            File::open(fixture_directory().join("dflash2_reference.safetensors"))
                .expect("the fixture opens"),
            None,
        )
        .expect("the fixture loads");
    let weights = bind_drafter_weights(&runtime, &checkpoint, &configuration)
        .expect("every drafter tensor binds");
    let fixture_tensor = |tensor_name: &str| {
        checkpoint
            .tensor(tensor_name)
            .expect("the fixture tensor exists")
    };
    let target = FixtureTarget {
        embedding_table: fixture_tensor("target.embed_tokens"),
        transposed_head: runtime
            .transpose_axes(&fixture_tensor("target.lm_head"), &[1, 0])
            .expect("the head transposes"),
    };
    let mut context_states = DrafterContextStates::new(&configuration, 0);
    for round_name in ["first_round", "second_round"] {
        let oracle =
            |output_name: &str| fixture_tensor(&format!("oracle.{round_name}.{output_name}"));
        let expected_position = values(&runtime, &oracle("context_position_before"))[0] as i32;
        assert_eq!(
            context_states.layers[0].next_context_position(),
            expected_position,
            "{round_name} context position"
        );
        let block_ids = oracle("block_ids");
        let hidden_rows = drafted_hidden_rows(
            &runtime,
            &weights,
            &configuration,
            &target,
            &mut context_states,
            &block_ids,
            &oracle("context_rows"),
        )
        .expect("the drafter runs");
        assert_close(
            &runtime,
            &hidden_rows,
            &oracle("final_hidden"),
            &format!("{round_name} final hidden rows"),
        );
        let logits = target
            .head_logits(&runtime, &hidden_rows)
            .expect("the head runs");
        assert_close(
            &runtime,
            &logits,
            &oracle("logits"),
            &format!("{round_name} logits"),
        );
        let anchor = runtime
            .reshape(
                &runtime
                    .slice(&block_ids, &[0, 0], &[1, 1], &[1, 1])
                    .expect("the anchor slices"),
                &[1],
            )
            .expect("the anchor reshapes");
        let path = select_path(
            &runtime,
            &weights.candidate_selector,
            &configuration,
            &hidden_rows,
            &logits,
            &anchor,
            &mut PathChoice::Greedy,
        )
        .expect("the selector runs");
        assert_eq!(
            values(&runtime, &path.token_ids),
            values(&runtime, &oracle("greedy_path")),
            "{round_name} greedy path"
        );
        assert_eq!(
            sorted_candidates_per_slot(&runtime, &path.candidate_ids),
            sorted_candidates_per_slot(&runtime, &oracle("candidates")),
            "{round_name} kept candidates"
        );
    }
}
