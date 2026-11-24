//! Unit contracts for the boundary checkpoint collector's per-layer records:
//! a layer records once, and forgetting it lets a redone layer record again.

use super::Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector;
use mlx_bridge::MlxArray;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

const BOUNDARY_TOKENS: [usize; 2] = [1_792, 3_584];
const BLOCK_TOKEN_COUNT: usize = 1_792;
const LAYER_INDEX: usize = 3;

fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn boundary_states(first_value: u32) -> Vec<MlxArray> {
    (0..BOUNDARY_TOKENS.len() as u32)
        .map(|boundary_index| {
            MlxArray::from_u32(&[first_value + boundary_index], &[1])
                .expect("a boundary state should upload")
        })
        .collect()
}

/// A collector for one linear-attention layer: two tensors per boundary.
fn collector() -> Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector {
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector::new(
        BOUNDARY_TOKENS.to_vec(),
        2,
        BLOCK_TOKEN_COUNT,
    )
    .expect("the collector should open")
}

#[test]
#[file_serial]
fn should_record_a_forgotten_layer_again_with_the_redone_values() {
    let _runtime = initialize_test_runtime();
    let mut collector = collector();
    collector
        .record_linear_attention_layer(LAYER_INDEX, boundary_states(10), boundary_states(20))
        .expect("the layer should record");
    collector.forget_linear_attention_layer(LAYER_INDEX);
    assert!(collector.layer_evaluation_arrays(LAYER_INDEX).is_empty());
    collector
        .record_linear_attention_layer(LAYER_INDEX, boundary_states(30), boundary_states(40))
        .expect("a forgotten layer should record again");
    let checkpoints = collector
        .complete()
        .expect("every boundary should hold the full layout");
    let recorded_convolution_states: Vec<Vec<u32>> = checkpoints
        .iter()
        .map(|checkpoint| {
            checkpoint.recurrent_snapshot_tensors
                [&format!("layer_{LAYER_INDEX}_linear.convolution")]
                .to_vec_u32()
                .expect("the recorded state should read back")
        })
        .collect();
    assert_eq!(recorded_convolution_states, vec![vec![30], vec![31]]);
}

#[test]
#[file_serial]
fn should_refuse_to_record_a_layer_twice_without_forgetting_it() {
    let _runtime = initialize_test_runtime();
    let mut collector = collector();
    collector
        .record_linear_attention_layer(LAYER_INDEX, boundary_states(10), boundary_states(20))
        .expect("the layer should record");
    assert!(
        collector
            .record_linear_attention_layer(LAYER_INDEX, boundary_states(30), boundary_states(40))
            .is_err()
    );
}

#[test]
#[file_serial]
fn should_leave_other_layers_alone_when_forgetting_one_that_recorded_nothing() {
    let _runtime = initialize_test_runtime();
    let mut collector = collector();
    collector
        .record_linear_attention_layer(LAYER_INDEX, boundary_states(10), boundary_states(20))
        .expect("the layer should record");
    collector.forget_linear_attention_layer(LAYER_INDEX + 1);
    assert_eq!(
        collector.layer_evaluation_arrays(LAYER_INDEX).len(),
        2 * BOUNDARY_TOKENS.len()
    );
    assert!(collector.complete().is_ok());
}
