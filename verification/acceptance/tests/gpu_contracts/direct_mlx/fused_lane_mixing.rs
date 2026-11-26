//! Flash-Next's fused lane-mixing kernels against the composed steps they
//! replace: the mix gate, the write-back weights and the write-back must
//! give the same bits, for one reply token and for a drafted block's rows.
//!
//! Random bfloat16 operands at Flash-Next's widths (four lanes of 2,560)
//! and at a small width that leaves a threadgroup partly empty, spread
//! near zero and far past where a sigmoid saturates. No model on
//! disk. Run with `cargo test-fused-lane-mixing`. Seconds.

use model_serving::qwen4_exp::hyper_connection::fused_lane_mixing_matches_the_composed_steps;
use model_serving::qwen4_exp::lane_mixing_kernels::LaneMixingKernels;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

fn kernel_runtime() -> MlxRuntime {
    let two_gigabytes = 2 * 1024 * 1024 * 1024;
    let memory_limits = MlxMemoryLimits::new(two_gigabytes, two_gigabytes / 4);
    MlxRuntime::initialize(memory_limits.expect("the limits are valid")).expect("MLX starts")
}

#[test]
#[ignore = "runs custom Metal kernels on the local GPU"]
#[serial]
fn should_match_the_composed_lane_mixing_steps_bit_for_bit_at_flash_next_widths() {
    let matches = fused_lane_mixing_matches_the_composed_steps(&kernel_runtime(), 4, 2_560)
        .expect("the kernels run");
    assert!(
        matches,
        "a fused lane-mixing kernel differs from the composed steps"
    );
}

#[test]
#[ignore = "runs custom Metal kernels on the local GPU"]
#[serial]
fn should_match_the_composed_steps_when_a_row_is_shorter_than_a_threadgroup() {
    let matches = fused_lane_mixing_matches_the_composed_steps(&kernel_runtime(), 2, 5)
        .expect("the kernels run");
    assert!(matches, "a fused lane-mixing kernel differs on a short row");
}

#[test]
fn should_declare_no_kernels_for_a_lane_count_whose_inverse_is_not_exact() {
    for lane_count in [0, -4, 3, 6] {
        let kernels = LaneMixingKernels::declare(lane_count).expect("declaring cannot fail");
        assert!(
            kernels.is_none(),
            "{lane_count} lanes must keep the composed steps"
        );
    }
}
