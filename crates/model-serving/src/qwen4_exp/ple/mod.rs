//! N-gram embedding table (PLE) concerns for `qwen4_exp`.
//!
//! Row identity is the only owner here so far: it turns token windows into
//! logical row numbers. Storage layout, shard mapping, byte ranges, row
//! caching, and quantized row decoding arrive with the artifact and streaming
//! steps and stay separate owners, so the rule and its storage can evolve
//! independently.

pub mod ngram_identity;

pub use ngram_identity::{
    DEFAULT_NGRAM_SEED, NgramIdentityConfiguration, NgramIdentityError, NgramPlanError,
    NgramRowIdentity, NgramVocabLayout, PLE_LAYER_PRIME, SPLITMIX_GAMMA, SPLITMIX_MULTIPLIER_1,
    SPLITMIX_MULTIPLIER_2, head_count_for, is_prime_u64, nth_prime_after, splitmix64,
};
