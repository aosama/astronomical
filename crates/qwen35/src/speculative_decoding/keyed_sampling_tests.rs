use super::dflash2_drafter::KeyedNoise;
use super::keyed_sampling::{RequestKeyedNoise, keyed_gumbel, keyed_sample, keyed_uniform};
use super::verification_decision::TargetDistribution;

#[test]
fn should_draw_the_uniforms_tensorfolds_keyed_sampler_draws() {
    // Values from a direct transcription of TensorFold's `uniform` in Python.
    assert_eq!(keyed_uniform(1234, 0, 0), 0.17704706673460952);
    assert_eq!(keyed_uniform(1234, 10, 42), 0.8056960556036823);
    assert_eq!(keyed_uniform(7, 248_000, 151_645), 0.12091485070483737);
}

#[test]
fn should_give_each_seed_position_and_token_its_own_draw() {
    let base = keyed_uniform(1, 2, 3);
    assert_ne!(base, keyed_uniform(2, 2, 3));
    assert_ne!(base, keyed_uniform(1, 3, 3));
    assert_ne!(base, keyed_uniform(1, 2, 4));
    assert_eq!(base, keyed_uniform(1, 2, 3));
}

#[test]
fn should_keep_uniforms_strictly_inside_zero_to_one_so_gumbel_stays_finite() {
    for position in 0..10_000 {
        let uniform = keyed_uniform(99, position, 5);
        assert!(uniform > 0.0 && uniform <= 1.0);
        assert!(keyed_gumbel(99, position, 5).is_finite());
    }
}

#[test]
fn should_sample_each_token_as_often_as_its_probability_across_positions() {
    let token_ids = [3_u32, 900, 70_000, 248_000];
    let probabilities = [0.1_f32, 0.2, 0.3, 0.4];
    let distribution = TargetDistribution {
        token_ids: &token_ids,
        probabilities: &probabilities,
    };
    let draw_count = 200_000;
    let mut counts = [0_u32; 4];
    for position in 0..draw_count {
        let sampled = keyed_sample(20_261_116, position, &distribution).expect("a token");
        let index = token_ids
            .iter()
            .position(|token_id| *token_id == sampled)
            .expect("a listed token");
        counts[index] += 1;
    }
    for (index, probability) in probabilities.iter().enumerate() {
        let observed = f64::from(counts[index]) / draw_count as f64;
        assert!(
            (observed - f64::from(*probability)).abs() < 0.005,
            "token {}: observed {observed}, expected {probability}",
            token_ids[index]
        );
    }
}

#[test]
fn should_never_sample_a_token_the_distribution_gives_no_probability() {
    let distribution = TargetDistribution {
        token_ids: &[1, 2, 3],
        probabilities: &[0.0, 1.0, 0.0],
    };
    for position in 0..1_000 {
        assert_eq!(keyed_sample(5, position, &distribution), Some(2));
    }
    let empty = TargetDistribution {
        token_ids: &[1, 2],
        probabilities: &[0.0, 0.0],
    };
    assert_eq!(keyed_sample(5, 0, &empty), None);
}

#[test]
fn should_lend_the_tree_the_same_draws_the_sampler_uses() {
    let noise = RequestKeyedNoise { seed: 42 };
    assert_eq!(noise.draw(17, 9_000), keyed_gumbel(42, 17, 9_000));
}
