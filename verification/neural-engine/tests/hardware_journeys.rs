//! Neural Engine journeys, by what they need.
//!
//! The founding operation is the `qwen3_5` MoE mixer output projection (DeltaNet
//! `out_proj` / attention `o_proj`) for one 2,048-token prefill chunk: 4,096 →
//! 2,048 channels, 34.4 GFLOP. Each hardware journey loads a program through public
//! Core ML and records where `MLComputePlan` placed it, how far its numbers sit
//! from the Accelerate fp32 reference, and how long one prediction takes.
//!
//! - `integration`: Core ML and MLX contracts on the CPU, with no Neural Engine and
//!   no real model. These are not ignored, so the integration lane runs them.
//! - `acceptance`: `#[ignore]`d contracts on real hardware that serving relies on.
//! - `measurement`: `#[ignore]`d journeys whose main output is the numbers they print.
//!
//! Run the ignored ones with `cargo test-neural-engine-journeys`, which gives each
//! its own process so an earlier load cannot blur a memory reading. Narrow with
//! `--filter`:
//!
//! - `deltanet`: the split journeys on the DeltaNet projections, running one
//!   projection's output channels on both engines at once.
//! - `compressed_weight`: the probe matrix (shift-scale and lookup-table weights, 4
//!   and 8 bits, block shapes) with placement, speed, and fidelity to the 4-bit
//!   checkpoint. `footprint_of_mlp_gate_and_up_as_<form>` measures one
//!   100-million-weight load's process-footprint growth.
//! - `discarding_the_compiled_model`: a program keeps predicting after its compiled
//!   copy leaves the temporary directory, which serving relies on.
//! - `model_hand_off`: one whole two-engine call per projection, with the
//!   activation conversion, the host copy, and the merge; about three minutes.
//! - `whole_mlp_split_by_token`: a whole dense MLP on the Neural Engine for a slice
//!   of tokens; half a minute. `one_mlp_program_per_layer` adds memory and cycling
//!   speed across distinct programs: about 90 seconds and roughly 18 GiB at its
//!   peak, and it stops loading at half of the Mac's memory.
//! - `one_real_ternary_mlp_layer`: placement, speed, and accuracy against the GPU on
//!   real Ternary-Bonsai-2-27B weights; about 15 seconds per weight form.
//!   `memory_of_real_ternary_mlp_programs` measures eight programs; about 80 seconds
//!   per form. `load_every_real_ternary_mlp_program` loads all 64 layers: about 11
//!   minutes and over 20 GiB, and it stops early if memory or disk runs low. All of
//!   these need `--configuration PATH` naming the checkpoint as `model_directory`.

mod acceptance;
mod integration;
mod measurement;
mod support;
