//! The one lifecycle vocabulary for every memory question in this package.
//!
//! `MemoryPhase` answers "where in the request lifecycle does this memory
//! question arise". Before this type existed, budgeting, growth learning, and
//! residency planning each carried a private phase enum with mismatched
//! variants; a reader had to reconcile all three to follow one request.
//!
//! Mapping contract (verified at the family mapping sites):
//!
//! - Budget composition (`budget/ram.rs`) treats `GenerationPreparation` as
//!   `Decode`-equivalent: the qwen3.5 residency planner maps
//!   `GenerationPreparation | Decode` to the decode budget, and laguna does
//!   the same.
//! - The adaptive growth guard only ever observes `Prefill` and `Decode`
//!   windows; callers never construct a growth context for
//!   `GenerationPreparation`.

/// Request lifecycle position a memory decision is made for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryPhase {
    /// Prompt chunks are being processed; activations are large and chunk-shaped.
    Prefill,
    /// The prompt finished; resident expert ownership is prepared before the
    /// first generated token. Budgeted as `Decode` because token writing is
    /// activation-cheap, but kept distinct so residency plans can preserve
    /// more expert payload before generation begins.
    GenerationPreparation,
    /// Tokens are being generated one at a time.
    Decode,
    /// No request work is in flight.
    Idle,
}
