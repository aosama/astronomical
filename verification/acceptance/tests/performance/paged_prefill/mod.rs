//! Prompt-processing benches for model weights paged from SSD.
//!
//! Cold-prefill measurements and the fixed-chunk sweep share this module: both
//! drive the in-process worker while the resident sparse model streams routed
//! experts from SSD, so every number here is a paged-prefill number.

// On-demand REST measurements that need the serving-side support modules,
// which only compile with the expert-paging feature.
#[cfg(feature = "expert-paging")]
mod back_to_back_residency_measurements;
mod cold_prefill_measurements;
#[cfg(feature = "expert-paging")]
mod prefill_chunk_restreaming_measurements;
mod prefill_chunk_sweep;
