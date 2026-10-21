//! Expert-paging REST journeys: one binary for the expert-paging capability.

#[cfg(feature = "expert-paging")]
#[path = "expert_paging/journeys/mod.rs"]
mod journeys;
#[cfg(feature = "expert-paging")]
mod support;
