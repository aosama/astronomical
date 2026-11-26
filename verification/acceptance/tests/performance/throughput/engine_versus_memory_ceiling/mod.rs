//! Throughput against the assigned MLX memory ceiling for the 35B sparse-MoE
//! model.
//!
//! One engine serves every text model: it keeps in memory what the ceiling
//! holds and streams the rest from the SSD. These cells show what each
//! gigabyte taken from the ceiling costs, with one identical workload, a
//! ~5,000-token Romeo and Juliet summarization into 300 output tokens,
//! `astro_ssd_prompt_cache` disabled:
//!
//! - `automatic_engine_ceiling_sweep`: startup at 24, 22, 21, 20, and 19 GB.
//!   It shows what a user gets as the ceiling shrinks and how throughput
//!   degrades as fewer expert layers stay seated.
//!
//! The 5,000/300 shape is a directed deviation from the canonical
//! 10,000/500 text series, recorded under its own journey kind so the
//! canonical series stays clean. Gigabytes follow the Apple marketing
//! convention: 1 GB = 1,073,741,824 bytes (2^30).
//!
//! Every cell loads real weights into wired GPU memory, so each is
//! `#[ignore]`d, `#[serial]`, and not wired into CI. Run them through the
//! test runner, which builds the exact optimized worker itself:
//! - all cells: `cargo test-engine-versus-memory-ceiling`
//! - add `--configuration PATH` with `{"diagnostics": true}` for per-operation
//!   performance attribution in the evidence directory.

mod automatic_engine_ceiling_sweep;
mod summarization_journey;
