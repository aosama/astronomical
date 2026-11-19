//! The top-k-only distribution against the sampler's full-vocabulary law it
//! must agree with: same kept tokens, same probabilities.

use mlx_bridge::MlxDtype;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use super::super::{masked_logits_after_top_k_and_top_p, softmax_last_axis};
use super::sampling_distribution_rows;

const VOCABULARY_SIZE: i32 = 4_096;
const ROW_COUNT: i32 = 6;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 256 * 1024 * 1024)
            .expect("the sampling test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// For each row, token id → probability under the full-vocabulary law.
fn full_vocabulary_law(
    runtime: &MlxRuntime,
    logits: &mlx_bridge::MlxArray,
    temperature_thousandths: u16,
    top_p_thousandths: u16,
    top_k: u16,
) -> Vec<f32> {
    let masked = masked_logits_after_top_k_and_top_p(
        runtime,
        logits,
        VOCABULARY_SIZE,
        top_k,
        top_p_thousandths,
    )
    .expect("the mask applies");
    let tempered = runtime
        .multiply_scalar(&masked, 1_000.0 / f32::from(temperature_thousandths))
        .expect("the temperature applies");
    let probabilities = softmax_last_axis(runtime, &tempered).expect("the softmax applies");
    runtime
        .array_to_vec_f32(
            &runtime
                .astype(&probabilities, MlxDtype::Float32)
                .expect("widen"),
        )
        .expect("probabilities read back")
}

#[test]
#[serial]
fn should_keep_the_tokens_and_probabilities_of_the_full_vocabulary_law() {
    let runtime = test_runtime();
    let key = runtime.random_key(17).expect("a key builds");
    let logits = runtime
        .random_normal(
            &[1, ROW_COUNT, VOCABULARY_SIZE],
            MlxDtype::BFloat16,
            0.0,
            3.0,
            &key,
        )
        .expect("logits build");
    for (temperature_thousandths, top_p_thousandths, top_k) in [
        (700, 950, 20),
        (1_000, 950, 20),
        (1_000, 1_000, 20),
        (300, 500, 5),
    ] {
        let (token_ids, probabilities) = sampling_distribution_rows(
            &runtime,
            &logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
        )
        .expect("the distribution builds");
        let kept_ids = token_ids.to_vec_u32().expect("ids read back");
        let kept_probabilities = runtime
            .array_to_vec_f32(&probabilities)
            .expect("probabilities read back");
        let law = full_vocabulary_law(
            &runtime,
            &logits,
            temperature_thousandths,
            top_p_thousandths,
            top_k,
        );
        let kept_per_row = usize::from(top_k);
        for row in 0..ROW_COUNT as usize {
            let mut kept_mass = 0.0;
            for slot in 0..kept_per_row {
                let token_id = kept_ids[row * kept_per_row + slot] as usize;
                let expected = law[row * VOCABULARY_SIZE as usize + token_id];
                let actual = kept_probabilities[row * kept_per_row + slot];
                // Both end in the logits' bf16 softmax, summed over the kept
                // tokens here and over the masked vocabulary there.
                assert!(
                    (expected - actual).abs() <= expected.max(1e-6) * 8e-3,
                    "row {row} token {token_id}: {actual} against the law's {expected}"
                );
                kept_mass += expected;
            }
            assert!(
                (kept_mass - 1.0).abs() <= 2e-2,
                "row {row}: the kept tokens carry the law's whole mass, not {kept_mass}"
            );
        }
    }
}
