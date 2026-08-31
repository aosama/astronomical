use astronomical_model_serving::qwen3_5_apply_top_p_mask;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_keep_the_candidate_that_crosses_the_top_p_threshold() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the sampler test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let mut candidate_probabilities = vec![0.0_f32; 20];
    candidate_probabilities[0] = 0.5;
    candidate_probabilities[1] = 0.4;
    candidate_probabilities[2] = 0.1;
    let candidate_logits = candidate_probabilities
        .iter()
        .map(|probability| probability.ln())
        .collect::<Vec<_>>();
    let probability_array = runtime
        .array_from_f32(&candidate_probabilities, &[1, 1, 20])
        .expect("the candidate probabilities should be valid");
    let logit_array = runtime
        .array_from_f32(&candidate_logits, &[1, 1, 20])
        .expect("the candidate logits should be valid");

    let masked_logits = qwen3_5_apply_top_p_mask(&runtime, &probability_array, &logit_array, 600)
        .expect("top-p filtering should build a valid MLX graph");
    let masked_logit_values = masked_logits
        .to_vec_f32()
        .expect("the masked logits should evaluate as float32");

    assert_eq!(masked_logit_values[0], candidate_logits[0]);
    assert_eq!(masked_logit_values[1], candidate_logits[1]);
    assert!(
        masked_logit_values[2..]
            .iter()
            .all(|logit| logit.is_infinite() && logit.is_sign_negative())
    );
}

#[tokio::test]
async fn should_reject_zero_mass_when_sampling_the_residual_correction() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the sampler test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    // Residual mass max(0, p - q) with a zero-mass token that must never emit.
    let residual_probabilities = vec![0.2_f32, 0.0, 0.5];
    let residuals = runtime
        .array_from_f32(&residual_probabilities, &[1, 1, 3])
        .expect("the residual probabilities should be valid");
    let emission_counts = sampled_residual_token_counts(&runtime, &residuals, 240);

    assert_eq!(emission_counts[1], 0, "a zero-mass token must never emit");
    assert!(
        emission_counts[0] > 0 && emission_counts[2] > 0,
        "both positive-mass tokens must be reachable: {emission_counts:?}"
    );
    // ln(0.2) vs ln(0.5): the sampled ratio must track the 2:5 mass ratio, not
    // an exp distortion of the raw masses.
    let token_zero_share =
        emission_counts[0] as f64 / (emission_counts[0] + emission_counts[2]) as f64;
    assert!(
        (0.2..=0.37).contains(&token_zero_share),
        "the sampled residual share must follow max(0, p - q): share {token_zero_share:.3}, counts {emission_counts:?}"
    );
}

#[tokio::test]
async fn should_draw_acceptance_coins_at_their_probabilities() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the sampler test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    // One depth-three proposal with probabilities 0.0, 1.0, and 0.5.
    let acceptance_probabilities = runtime
        .array_from_f32(&[0.0_f32, 1.0, 0.5], &[1, 1, 3])
        .expect("the acceptance probabilities should be valid");
    // Deterministic cases first: probability zero always rejects, one always accepts.
    let deterministic = runtime
        .array_from_f32(&[0.0_f32, 1.0], &[1, 1, 2])
        .expect("the deterministic acceptance probabilities should be valid");
    for _attempt in 0..10 {
        let random_state = runtime
            .random_key(u64::from_le_bytes([7_u8; 8]))
            .expect("the random key should initialize");
        let mut random_state = random_state;
        let coins = astronomical_model_serving::sample_acceptance_coins_for_tests(
            &runtime,
            &deterministic,
            &mut random_state,
            2,
        )
        .expect("the acceptance coins should sample");
        let coin_values = coins.to_vec_u32().expect("the coins should evaluate");
        assert_eq!(coin_values[0], 0, "probability zero must always reject");
        assert_eq!(coin_values[1], 1, "probability one must always accept");
    }
    let _ = acceptance_probabilities;
    // The half-probability case must reach both outcomes across draws.
    let half_probabilities = runtime
        .array_from_f32(&[0.5_f32], &[1, 1, 1])
        .expect("the half acceptance probability should be valid");
    let mut accept_count = 0_u32;
    for draw_index in 0..64_u64 {
        let random_state = runtime
            .random_key(1_000 + draw_index)
            .expect("the random key should initialize");
        let mut random_state = random_state;
        let coins = astronomical_model_serving::sample_acceptance_coins_for_tests(
            &runtime,
            &half_probabilities,
            &mut random_state,
            1,
        )
        .expect("the acceptance coin should sample");
        if coins.to_vec_u32().expect("the coin should evaluate")[0] == 1 {
            accept_count += 1;
        }
    }
    assert!(
        accept_count > 8 && accept_count < 56,
        "a half-probability coin must land on both outcomes across draws: {accept_count}"
    );
}

fn sampled_residual_token_counts(
    runtime: &MlxRuntime,
    residuals: &astronomical_runtime_integration::MlxArray,
    draw_count: usize,
) -> Vec<u32> {
    let mut emission_counts = vec![0_u32; 3];
    for draw_index in 0..draw_count as u64 {
        let random_state = runtime
            .random_key(500 + draw_index)
            .expect("the random key should initialize");
        let mut random_state = random_state;
        let sampled_token =
            astronomical_model_serving::sample_from_relative_probabilities_for_tests(
                runtime,
                residuals,
                &mut random_state,
            )
            .expect("the residual sample should build and evaluate");
        let token_id = sampled_token
            .item_u32()
            .expect("the residual sample should be one token");
        emission_counts[token_id as usize] += 1;
    }
    emission_counts
}
