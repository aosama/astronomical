//! The one lifecycle vocabulary for every memory question in this package.
//!
//! `MemoryPhase` answers "where in the request lifecycle does this memory
//! question arise". The residency planner treats `GenerationPreparation`
//! like `Decode` when it sizes a reply's plan. The memory ledger has its
//! own, smaller phase type (`ledger::LedgerPhase`), because all it needs to
//! know is whether a prompt is being read, a reply written, or neither.

/// Request lifecycle position a memory decision is made for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MemoryPhase {
    /// Prompt chunks are being processed; activations are large and chunk-shaped.
    Prefill,
    /// The prompt finished; resident expert ownership is prepared before the
    /// first generated token. Planned like `Decode` because writing a token
    /// needs few MLX working buffers, but kept distinct so residency plans
    /// can preserve more expert payload before generation begins.
    GenerationPreparation,
    /// Tokens are being generated one at a time.
    Decode,
    /// No request work is in flight.
    Idle,
}
