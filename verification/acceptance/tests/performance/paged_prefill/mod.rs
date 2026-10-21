//! Prompt-processing benches for model weights paged from SSD.
//!
//! Cold-prefill measurements and the fixed-chunk sweep share this module: both
//! drive the in-process worker while the resident sparse model streams routed
//! experts from SSD, so every number here is a paged-prefill number.

mod cold_prefill_measurements;
mod prefill_chunk_sweep;
