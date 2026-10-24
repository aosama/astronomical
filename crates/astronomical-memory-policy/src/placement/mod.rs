//! The typed engine-placement lifecycle: why the loaded engine was placed,
//! why a request demoted it, and when a demoted engine may return to the
//! complete-residency resident engine.

mod engine_placement;
mod resident_promotion;

#[cfg(test)]
mod engine_placement_tests;
#[cfg(test)]
mod resident_promotion_tests;

pub use engine_placement::EnginePlacement;
pub use resident_promotion::{
    ResidentPromotionBlocker, ResidentPromotionDecision,
    should_promote_demoted_streaming_engine_to_resident,
};
