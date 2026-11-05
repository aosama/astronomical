//! Real-hardware Neural Engine journeys.
//!
//! The founding operation is the `qwen3_5` MoE mixer output projection (DeltaNet
//! `out_proj` / attention `o_proj`) for one 2,048-token prefill chunk: 4,096 →
//! 2,048 channels, 34.4 GFLOP. Each journey loads a program through public Core ML
//! and records where `MLComputePlan` placed it, how far its numbers sit from the
//! Accelerate fp32 reference, and how long one prediction takes.
//!
//! `acceptance` holds journeys that assert a contract serving relies on, and
//! `measurement` holds journeys whose main output is the numbers they report.
//!
//! Run with `cargo test-neural-engine-journeys` (optimized build, one process per
//! journey); the ternary journeys also need `--configuration PATH`.

mod acceptance;
mod measurement;
mod support;
