//! The top-k kernel against a full sort on the host.

use mlx_bridge::MlxDtype;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::vocabulary_top_k;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(1024 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("the top-k test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// Each row's `kept_count` best token ids by a host sort: highest logit
/// first, the lower id first among equal logits.
fn host_top_k(values: &[f32], vocabulary_size: usize, kept_count: usize) -> Vec<u32> {
    let mut expected = Vec::new();
    for row in values.chunks(vocabulary_size) {
        let mut order: Vec<usize> = (0..vocabulary_size).collect();
        order.sort_by(|left, right| row[*right].total_cmp(&row[*left]).then(left.cmp(right)));
        expected.extend(order[..kept_count].iter().map(|token| *token as u32));
    }
    expected
}

#[test]
#[file_serial]
fn should_find_each_rows_best_tokens_in_order_as_a_full_sort_does() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_find_each_rows_best_tokens_in_order_as_a_full_sort_does",
    ) {
        return;
    }
    let runtime = test_runtime();
    let key = runtime.random_key(3).expect("a key builds");
    // A real vocabulary width with bf16 logits, which tie often.
    for (row_count, vocabulary_size, kept_count) in [
        (16, 248_320, 20),
        (7, 248_320, 16),
        (3, 4_096, 32),
        (2, 50, 1),
    ] {
        let logits = runtime
            .random_normal(
                &[1, row_count, vocabulary_size],
                MlxDtype::BFloat16,
                0.0,
                3.0,
                &key,
            )
            .expect("logits build");
        let token_ids = vocabulary_top_k(&runtime, &logits, kept_count).expect("the kernel runs");
        let host_values = runtime
            .array_to_vec_f32(&runtime.astype(&logits, MlxDtype::Float32).expect("widen"))
            .expect("logits read back");
        assert_eq!(
            token_ids.to_vec_u32().expect("ids read back"),
            host_top_k(&host_values, vocabulary_size as usize, kept_count as usize),
            "{row_count} rows of {vocabulary_size}, top {kept_count}"
        );
    }
}

#[test]
#[file_serial]
fn should_refuse_more_than_32_tokens_a_row_or_none() {
    let runtime = test_runtime();
    let logits = runtime
        .zeros(&[1, 2, 100], MlxDtype::BFloat16)
        .expect("logits build");
    assert!(vocabulary_top_k(&runtime, &logits, 33).is_err());
    assert!(vocabulary_top_k(&runtime, &logits, 0).is_err());
}
