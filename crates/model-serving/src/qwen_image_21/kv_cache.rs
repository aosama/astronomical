//! Per-layer KV-cache control logic for Qwen-Image-2.1.
//!
//! The denoise loop runs the transformer once per step, but everything that is *not* the target
//! image is identical every step: the text/condition prefix does not change, and `causal_condition`
//! modulation gives every non-target token the shared `t = 0` row. So the intended scheme runs the
//! whole joint sequence exactly once in `extract` mode, stores the prefix keys/values, and every
//! later step runs only the target-image queries against that cached prefix.
//!
//! Status: the transformer's runtime path currently runs the reference's non-cached forward
//! (`kv_cache = None`), matching the pipeline default here. This module is the verified bookkeeping
//! for the cached path — it is what `prefix_length` already supplies, and what a cache-carrying
//! forward would need — so it is hermetically tested and kept, not dead residue. Do not read its
//! presence as evidence that the runtime caches prefix keys and values yet.
//!
//! The whole scheme rests on one layout fact: the target-image tokens must be the **contiguous
//! suffix** of the joint sequence. The forward slices `joint_hidden_states[:, prefix_len:]` and
//! caches `[0, prefix_len)` as single ranges, which is only correct under that invariant, so
//! [`cache_is_valid`] checks it (and the `causal_condition` gate the reference requires) before the
//! runtime relies on it. This module is pure integer/tuple bookkeeping — no runtime dependency — so
//! it is tested hermetically against the diffusers oracle.

/// Which caching mode a cache-carrying transformer forward would run for a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Warm-up without caching: process the whole joint sequence, keep nothing.
    Prefill,
    /// Populate the cache: process the whole joint sequence once and store the prefix keys/values.
    Extract,
    /// Consume the cache: process only the target-image queries against the cached prefix.
    Reuse,
}

/// Number of prefix (non-target) tokens: the count the reference computes as `(~mask).sum()`. This is
/// the cached length in `Extract` mode and the boundary between prefix and target everywhere else.
#[must_use]
pub fn prefix_length(target_token_mask: &[bool]) -> usize {
    target_token_mask
        .iter()
        .filter(|&&is_target| !is_target)
        .count()
}

/// Whether the joint-sequence layout is cacheable, i.e. the runtime may use [`CacheMode::Extract`]
/// then [`CacheMode::Reuse`] for this step.
///
/// True requires all of: caching is enabled (`causal_condition_enabled`, the reference gates the
/// whole KV-cache path on `causal_condition=True`), the target tokens form the contiguous suffix
/// `[prefix_len, seq_len)`, there is a non-empty prefix to reuse, and a non-empty target to decode.
/// A mixed/interleaved layout, a pure-target sequence, or a pure-text sequence all return `false`,
/// because in each case the single-range `[prefix_len:]` / `[0, prefix_len)` split would be wrong.
#[must_use]
pub fn cache_is_valid(target_token_mask: &[bool], causal_condition_enabled: bool) -> bool {
    if !causal_condition_enabled {
        return false;
    }
    let seq_len = target_token_mask.len();
    let prefix_len = prefix_length(target_token_mask);
    let target_count = seq_len - prefix_len;
    if target_count == 0 || target_count == seq_len {
        return false;
    }
    let prefix_all_non_target = target_token_mask[..prefix_len]
        .iter()
        .all(|&is_target| !is_target);
    let target_all_target = target_token_mask[prefix_len..]
        .iter()
        .all(|&is_target| is_target);
    prefix_all_non_target && target_all_target
}

/// The half-open range of joint-sequence positions the forward computes for this step.
///
/// `Prefill` and `Extract` compute the whole sequence `[0, seq_len)`; `Reuse` computes only the
/// target queries `[prefix_len, seq_len)` and read the cached prefix for their keys/values.
#[must_use]
pub fn cache_query_slice(mode: CacheMode, seq_len: usize, prefix_len: usize) -> (usize, usize) {
    match mode {
        CacheMode::Prefill | CacheMode::Extract => (0, seq_len),
        CacheMode::Reuse => (prefix_len, seq_len),
    }
}

/// The half-open range to write into the cache, or `None` when the step does not populate it.
///
/// Only `Extract` writes, and it writes the prefix `[0, prefix_len)`; `Prefill` keeps nothing and
/// `Reuse` reads the values written by an earlier `Extract`.
#[must_use]
pub fn cache_write_slice(mode: CacheMode, prefix_len: usize) -> Option<(usize, usize)> {
    match mode {
        CacheMode::Extract => Some((0, prefix_len)),
        CacheMode::Prefill | CacheMode::Reuse => None,
    }
}
