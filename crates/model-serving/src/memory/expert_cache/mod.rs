//! Expert-cache policy owned by the memory package.
//!
//! Prefill pinning stays in `expert_paging::RetainedExpertPageCache`. Decode
//! residency lives in `decode` and never names a layer pin.

pub mod decode;

pub use decode::{DecodeExpertCache, ResidentExpertWeight};
