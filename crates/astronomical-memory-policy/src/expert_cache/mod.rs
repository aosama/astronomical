//! Expert-cache policy for decode residency, owned by the
//! `astronomical-memory-policy` crate.
//!
//! Prefill pinning stays in `astronomical-model-serving`'s
//! `expert_paging::RetainedExpertPageCache`. Decode residency lives in
//! `decode` and never names a layer pin; the direct-MLX page-materialization
//! arithmetic stays beside the bindings in the serving crate.

pub mod decode;
