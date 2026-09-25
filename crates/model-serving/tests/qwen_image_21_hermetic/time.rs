//! Hermetic tests for the Qwen-Image-2.1 sinusoidal timestep embedding, compared against the
//! reference frequency table and embeddings for four representative sigmas.

use astronomical_model_serving::SinusoidalTimesteps;
use serde_json::Value;

use super::support::{oracle_document, oracle_f32_array, oracle_f64_values};

/// Reference frequency table plus embeddings, derived from the diffusers
/// QwenImage21TemporalTimesteps implementation.
const TIME_ORACLE_JSON: &str = include_str!("../fixtures/qwen_image_21/time_oracle.json");

/// Slack for the f32 embedding: both sides compute cos/sin in f64 and cast, so the gap is sub-ULP.
const EMBED_TOLERANCE: f32 = 1e-5;
/// Slack for the f64 frequency table: a wrong exponent or half-width would be orders larger.
const FREQ_TOLERANCE: f64 = 1e-12;

fn time_oracle() -> Value {
    oracle_document(TIME_ORACLE_JSON)
}

fn timestep_dim(oracle: &Value) -> usize {
    oracle["timestep_dim"]
        .as_u64()
        .expect("timestep_dim must exist") as usize
}

fn timesteps(oracle: &Value) -> Vec<f64> {
    oracle_f64_values(oracle, "timesteps")
}

#[test]
fn should_match_the_oracle_frequency_table() {
    let projection = SinusoidalTimesteps::default();
    let oracle_freqs = oracle_f64_values(&time_oracle(), "freqs");
    let freqs = projection.freqs_for_tests();
    assert_eq!(
        freqs.len(),
        oracle_freqs.len(),
        "frequency table length must equal timestep_dim / 2"
    );
    for (index, (&computed, &oracle)) in freqs.iter().zip(oracle_freqs.iter()).enumerate() {
        let diff = (computed - oracle).abs();
        assert!(
            diff <= FREQ_TOLERANCE,
            "freq[{index}]={computed:?} oracle={oracle:?} (diff {diff})"
        );
    }
}

#[test]
fn should_match_the_diffusers_sinusoidal_embedding() {
    let projection = SinusoidalTimesteps::default();
    let oracle = time_oracle();
    let oracle_steps = timesteps(&oracle);
    let oracle_embeddings = oracle["oracle_embed"]
        .as_array()
        .expect("oracle_embed must be an array")
        .iter()
        .map(oracle_f32_array)
        .collect::<Vec<_>>();
    assert_eq!(oracle_steps.len(), oracle_embeddings.len());

    for (step, &timestep) in oracle_steps.iter().enumerate() {
        let embedding = projection.embed(timestep);
        assert_eq!(
            embedding.len(),
            timestep_dim(&oracle),
            "embedding width must equal timestep_dim"
        );
        assert_eq!(embedding.len(), oracle_embeddings[step].len());
        for (channel, &computed) in embedding.iter().enumerate() {
            let oracle_value = oracle_embeddings[step][channel];
            let diff = (computed - oracle_value).abs();
            assert!(
                diff <= EMBED_TOLERANCE,
                "timestep {timestep} channel {channel}: computed={computed:?} oracle={oracle_value:?} (diff {diff})"
            );
        }
    }
}

#[test]
fn should_emit_the_cosine_half_before_the_sine_half() {
    // At timestep 0 every argument is 0, so cos == 1 across the first half and sin == 0 across the
    // second half. This pins the channel order: swapping the halves would silently corrupt the
    // embedding that feeds the AdaLN modulation.
    let projection = SinusoidalTimesteps::default();
    let embedding = projection.embed(0.0);
    let half = timestep_dim(&time_oracle()) / 2;
    for channel in 0..half {
        assert!(
            (embedding[channel] - 1.0).abs() <= EMBED_TOLERANCE,
            "cos half channel {channel} must be 1.0 at t=0"
        );
    }
    for channel in half..embedding.len() {
        assert!(
            embedding[channel].abs() <= EMBED_TOLERANCE,
            "sin half channel {channel} must be 0.0 at t=0"
        );
    }
}

#[test]
fn should_pad_an_odd_dimension_with_a_single_zero_channel() {
    // The reference appends one zero channel when timestep_dim is odd; the even default must not.
    let odd = SinusoidalTimesteps::new(7, 10000.0, 1000.0);
    let embedding = odd.embed(0.5);
    assert_eq!(
        embedding.len(),
        7,
        "odd timestep_dim must yield timestep_dim channels"
    );
    assert_eq!(
        embedding[6], 0.0,
        "the trailing pad channel must be exactly zero"
    );

    let even = SinusoidalTimesteps::default();
    assert_eq!(
        even.embed(0.5).len(),
        timestep_dim(&time_oracle()),
        "even timestep_dim must not pad"
    );
}
