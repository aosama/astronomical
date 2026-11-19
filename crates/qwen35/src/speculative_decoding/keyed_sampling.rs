//! Keyed sampling: the random draw for a token at a position is a fixed hash
//! of (seed, absolute position, token id), so the token a request samples at
//! a position never depends on how the rows before it were drafted.
//!
//! Ported from TensorFold's keyed sampler (release 1.2.0,
//! `zig/src/core/lanes/sampling.zig`, Apache-2.0, Copyright 2026 TensorFold
//! contributors). A sample is the Gumbel-max of the target's distribution:
//! `argmax_t log p(t) + G(seed, position, t)` with `G = -ln(-ln u)` and `u`
//! uniform. With independent uniforms per token that is an exact draw from
//! `p`, whatever `p` is. Because the draws are fixed per position, every node
//! of a draft tree is checked by token equality alone, and the drafter can
//! add the same draws to its scores to lean toward what will be sampled.

use super::dflash2_drafter::KeyedNoise;
use super::verification_decision::TargetDistribution;

const SEED_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const POSITION_MULTIPLIER: u64 = 0xD1B5_4A32_D192_ED03;

/// splitmix64's finalizer.
fn mix(value: u64) -> u64 {
    let mut mixed = value;
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// A uniform in (0, 1] from the hash's top 53 bits, never exactly 0, so its
/// Gumbel transform is always finite.
pub fn keyed_uniform(seed: u64, absolute_position: u64, token_id: u32) -> f64 {
    let position_base = mix(mix(seed.wrapping_add(SEED_INCREMENT))
        ^ absolute_position.wrapping_mul(POSITION_MULTIPLIER));
    let bits = mix(position_base ^ u64::from(token_id));
    (bits >> 11) as f64 * 2f64.powi(-53) + 2f64.powi(-54)
}

/// The Gumbel draw `-ln(-ln u)` for a token at a position.
pub fn keyed_gumbel(seed: u64, absolute_position: u64, token_id: u32) -> f64 {
    -(-keyed_uniform(seed, absolute_position, token_id).ln()).ln()
}

/// The token a request with `seed` samples at `absolute_position` from
/// `distribution`: the Gumbel-max over the tokens it gives probability.
/// None only when every probability is zero.
pub fn keyed_sample(
    seed: u64,
    absolute_position: u64,
    distribution: &TargetDistribution<'_>,
) -> Option<u32> {
    let mut best: Option<(u32, f64)> = None;
    for (token_id, probability) in distribution
        .token_ids
        .iter()
        .zip(distribution.probabilities)
    {
        if *probability <= 0.0 {
            continue;
        }
        let score = f64::from(*probability).ln() + keyed_gumbel(seed, absolute_position, *token_id);
        let is_better = match best {
            None => true,
            Some((_, best_score)) => score > best_score,
        };
        if is_better {
            best = Some((*token_id, score));
        }
    }
    best.map(first_of_pair)
}

fn first_of_pair((token_id, _score): (u32, f64)) -> u32 {
    token_id
}

/// The keyed draws of one request, as the draft tree adds them to its scores.
#[derive(Debug, Clone, Copy)]
pub struct RequestKeyedNoise {
    pub seed: u64,
}

impl KeyedNoise for RequestKeyedNoise {
    fn draw(&self, absolute_position: u64, token_id: u32) -> f64 {
        keyed_gumbel(self.seed, absolute_position, token_id)
    }
}
