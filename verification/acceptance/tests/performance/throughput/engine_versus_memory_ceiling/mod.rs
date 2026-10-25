//! Engine-versus-memory-ceiling throughput comparison for the resident 35B
//! sparse-MoE model.
//!
//! Startup couples the assigned MLX memory ceiling to the engine family: a
//! ceiling that holds every expert selects the resident engine, a smaller one
//! selects the SSD-streaming engine. A throughput gap between two ceilings is
//! therefore ambiguous — it may be the ceiling squeezing memory or the engine
//! swap itself. These cells separate the two causes with one identical
//! workload, a ~5,000-token Romeo and Juliet summarization into 300 output
//! tokens, persistent prompt cache disabled, 2,048-token prefill chunks:
//!
//! - `automatic_engine_ceiling_sweep`: the production engine choice at 24, 22,
//!   21, 20, and 19 GB. It shows what a user gets as the ceiling shrinks and
//!   how throughput degrades as fewer expert layers stay seated.
//! - `forced_engine_isolation`: the streaming engine forced at roomy ceilings
//!   (its own cost with every expert seated and no disk reads) and the
//!   resident engine forced at a tight ceiling (what the ceiling alone costs).
//!   The 32 GB forced-streaming cell matching the 24 GB resident cell is the
//!   parity target for the streaming engine.
//!
//! The 5,000/300 shape is a directed deviation from the canonical
//! 10,000/500 text series, recorded under its own journey kind so the
//! canonical series stays clean. Only automatic-engine cells reach the durable
//! history; forced-engine cells print their record and evidence but are never
//! appended, because production never runs a forced engine. Gigabytes follow the Apple marketing
//! convention: 1 GB = 1,073,741,824 bytes (2^30).
//!
//! Every cell loads real weights into wired GPU memory, so each is
//! `#[ignore]`d, `#[serial]`, and not wired into CI. Run them through the
//! test runner, which builds the exact optimized worker itself:
//! - all cells: `cargo test-engine-versus-memory-ceiling`
//! - one group: `cargo xtask test-run --package astronomical-acceptance
//!   --features performance_throughput --test performance_throughput --release
//!   --filter engine_versus_memory_ceiling::forced_engine_isolation --ignored`
//! - add `--configuration PATH` with `{"diagnostics": true}` for per-operation
//!   performance attribution in the evidence directory.

mod automatic_engine_ceiling_sweep;
mod forced_engine_isolation;
mod summarization_journey;
