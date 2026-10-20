//! Sensitivity-driven mixed-precision affine quantization for MLX artifacts.
//!
//! This crate is a Rust port of the quantization core of OptiQ (the
//! `mlx-optiq` Python package, MIT License, Copyright (c) 2026 Thin Signal;
//! see `third-party/THIRD_PARTY_NOTICES` and `calibration/LICENSE-OPTIQ`).
//! The port keeps OptiQ's algorithms and its on-disk artifact format, so an
//! artifact produced here loads under stock mlx-lm, under OptiQ, and under
//! Astronomical's own strict OptiQ readers in the model-serving crate. The
//! reasons for porting rather than shelling out to Python are speed (the
//! tensor-at-a-time requantizer and the per-group range search are
//! embarrassingly parallel and bandwidth-bound) and control (Astronomical
//! wants to modify the allocation policy and the encoder without forking a
//! 93,000-line Python application).
//!
//! # Pipeline
//!
//! The pipeline mirrors OptiQ's `run_llm_pipeline`, one typed stage per module:
//!
//! 1. [`module_inventory`] reads the source safetensors headers and lists every
//!    group-quantizable module (2-D linears, 3-D fused experts) with its
//!    logical parameter count and transformer block index. No tensor data is
//!    read.
//! 2. [`calibration`] turns the bundled multi-domain calibration mix (or a
//!    caller-supplied JSONL file) into fixed-length token sequences.
//! 3. [`sensitivity`] scores every module at every candidate bit width: the
//!    exact sweep measures KL divergence through a [`sensitivity::SensitivityProbeModel`]
//!    (implemented by model-serving for Qwen3.5), and the static scorer ranks
//!    modules by architecture alone when a forward pass is too expensive.
//! 4. [`allocation`] runs the greedy knapsack with the block-aware floor and
//!    the block-run guard to turn the sensitivity table into one bit width per
//!    module under a bits-per-weight target.
//! 5. [`requantize`] walks the source shards tensor by tensor, encodes each
//!    weight at its allocated width with the range-search [`encoder`], and
//!    streams the result to disk with bounded memory.
//! 6. [`artifact`] writes the `config.json` quantization documents,
//!    `optiq/metadata.json`, `optiq/sensitivity.json`, and the shard index so
//!    the output is a complete, loadable artifact.
//!
//! [`latency`] (the roofline throughput model) and [`kv_cache`] (mixed-precision
//! key-value cache allocation) are ported alongside because they consume the
//! same sensitivity signal, but neither sits on the critical path of a
//! conversion.
//!
//! # Feature worlds
//!
//! Everything that needs a Metal device sits behind the `direct-mlx` feature:
//! the encoder, the requantizer, and the MLX implementation of KL divergence.
//! The rest of the crate is pure Rust, which keeps the allocator, the schemas,
//! the calibration loader, and the sweep driver testable in the offline lane
//! and usable from the CLI's `quantize plan` sub-verb on any host.
//!
//! # Progress reporting
//!
//! Every long-running stage (the sensitivity sweep, the requantizer, the
//! calibration tokenization) reports percentage completion and an estimated
//! time to completion through [`progress::ProgressReporter`] at a fixed
//! interval. This is a repository rule for batch jobs, not a nicety: a sweep
//! that streams bf16 weights from SSD runs for hours and a log that shows
//! nothing until exit is indistinguishable from a hang.

#![forbid(unsafe_code)]

pub mod allocation;
pub mod artifact;
pub mod bit_width;
pub mod calibration;
pub mod error;
pub mod kv_cache;
pub mod latency;
pub mod module_inventory;
pub mod module_taxonomy;
pub mod pipeline;
pub mod progress;
pub mod sensitivity;

pub mod encoder;
pub mod requantize;

pub use allocation::{
    AllocationConstraints, AllocationResult, ModuleQuantizationDecision, QuantizationPredicate,
    optimize_mixed_precision,
};
pub use artifact::{OptiQArtifactMetadata, OptiQSensitivityDocument, QuantizationConfigDocument};
pub use bit_width::{BitWidth, GroupSize, KEEP_BFLOAT16_BITS};
pub use calibration::{CalibrationMix, CalibrationSample, CalibrationTokenizer};
pub use error::QuantizationError;
pub use module_inventory::{ModuleInventory, QuantizableModuleDescriptor};
pub use module_taxonomy::ModuleRole;
pub use pipeline::{QuantizationPipelineOutcome, QuantizationPipelineRequest, ReferenceMode};
pub use progress::{EtaEstimate, ProgressReporter, StageProgress};
pub use sensitivity::{
    ModuleRestoreToken, SensitivityMethod, SensitivityProbeModel, SensitivityResult,
    SensitivityTable,
};
