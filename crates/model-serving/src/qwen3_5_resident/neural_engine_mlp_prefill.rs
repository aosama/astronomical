//! Opt-in prefill that runs a slice of each full chunk's tokens through whole
//! dense MLP programs on the Neural Engine, next to the GPU (issue #161). The
//! linear-attention output gates have their own programs in
//! `neural_engine_output_gate_prefill`.
//!
//! The MLP moves as one program because that needs one hand-off per layer, where
//! splitting single projections by channel needs one per projection. A whole-MLP
//! token split also measured 1.24× against 1.07× for moving only gate and up by
//! channel. Mixture-of-experts models are out of scope because their routed experts
//! stay on the GPU, which measured about 1.04× for the chunk.

mod program_loading;
mod token_split_forward;

pub(crate) use program_loading::{
    FP16_BYTES_PER_WEIGHT, MlpProgramPlan, plan_neural_engine_mlp_programs, program_surface_bytes,
    unavailable, weight_as_fp16_bits,
};

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{NeuralEngineChunkPlacement, NeuralEngineChunkReport};
use astronomical_memory_policy::neural_engine_admission::chunk_fits_beside_programs;
use astronomical_neural_engine::core_ml::CoreMlProgram;

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_chunk_comparison::{ChunkComparison, ChunkEngines};
use super::neural_engine_program_speed_record::ProgramSpeedRecord;

// A prediction takes milliseconds, so this only bounds a wedged Neural Engine.
pub(crate) const NEURAL_ENGINE_FINISH_TIMEOUT: Duration = Duration::from_secs(30);

// 47 percent: on a real mixed-bit artifact at 44 percent the GPU finished last in
// 88 percent of layers (18.6 ms against 16.5 ms), and each 64 tokens moves about
// 1 ms between the engines. The chunk's other half, attention and the mixer, runs
// on the GPU regardless, which is why real prefill through the daemon measured
// 1.13× to 1.2× on Ornith-1.5-9B, with the prompt cache on and off, and not more.
// A sweep kept it: a 2B model peaked at 704 to 960 tokens of 2,048 and fell below the
// GPU alone at 1,216, a 9B model gained less at 704, and no share helped a 0.8B model
// (experiments.md), which `neural_engine_chunk_comparison` turns off instead.
const NEURAL_ENGINE_SHARE_NUMERATOR: u32 = 15;
const NEURAL_ENGINE_SHARE_DENOMINATOR: u32 = 32;
// A surface row must fill its alignment exactly for MLX to read it as one array,
// and multiples of 64 fp16 tokens do.
pub(crate) const NEURAL_ENGINE_TOKEN_GRANULARITY: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkFitDecision {
    does_chunk_fit_memory: bool,
    has_changed: bool,
}

/// One program per decoder layer, each fixed to the chunk's token slice.
pub(crate) struct NeuralEngineMlpPrefill {
    layer_programs: RefCell<Vec<CoreMlProgram>>,
    chunk_token_count: u32,
    neural_engine_token_count: u32,
    /// Every program's cost on this Mac, the MLPs and the output gates: what their first
    /// runs wired, plus the surfaces wired before those runs.
    measured_program_bytes: u64,
    /// The surfaces alone, the part of that a reading of wired growth cannot see.
    #[cfg_attr(not(feature = "direct-mlx"), allow(dead_code))]
    surface_bytes: u64,
    /// Set by the first failure of a program, so later chunks stay on the GPU.
    pub(in crate::qwen3_5_resident) has_failed: Cell<bool>,
    /// Lets one loaded model alternate between arms in a measurement, because
    /// separate runs differ by warm-up and heat as much as the effects measured.
    is_paused: Cell<bool>,
    /// Decided before each chunk from what MLX is projected to peak at. The driver
    /// keeps the programs wired for 10 to 20 seconds after it turns false, so a chunk
    /// run then can briefly sit above the ceiling by what the GPU's share adds.
    does_chunk_fit_memory: Cell<bool>,
    /// Whether the coming chunk runs on the programs, from its fit and the comparison.
    does_chunk_use_programs: Cell<bool>,
    pub(in crate::qwen3_5_resident) chunk_comparison: RefCell<ChunkComparison>,
    /// Where the comparison's answer is kept, when this set has a folder to keep it in.
    pub(in crate::qwen3_5_resident) speed_record: Option<ProgramSpeedRecord>,
    /// A verdict reaches the status and the record once.
    pub(in crate::qwen3_5_resident) has_reported_verdict: Cell<bool>,
}

impl std::fmt::Debug for NeuralEngineMlpPrefill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NeuralEngineMlpPrefill")
            .field("chunk_token_count", &self.chunk_token_count)
            .field("neural_engine_token_count", &self.neural_engine_token_count)
            .field("has_failed", &self.has_failed.get())
            .finish_non_exhaustive()
    }
}

impl NeuralEngineMlpPrefill {
    /// Decides the coming chunk and says whether its fit changed. Only a full chunk that
    /// fits is put to the comparison, because no other can run on the programs.
    fn decide_chunk(
        &self,
        chunk_token_count: usize,
        projected_chunk_peak_bytes: u64,
        active_memory_ceiling_bytes: u64,
        now: Instant,
    ) -> ChunkFitDecision {
        let does_chunk_fit_memory = chunk_fits_beside_programs(
            projected_chunk_peak_bytes,
            self.measured_program_bytes,
            active_memory_ceiling_bytes,
        );
        let did_fit_before = self.does_chunk_fit_memory.replace(does_chunk_fit_memory);
        let is_full_chunk = u32::try_from(chunk_token_count) == Ok(self.chunk_token_count);
        self.does_chunk_use_programs.set(
            does_chunk_fit_memory
                && is_full_chunk
                && self.chunk_comparison.borrow_mut().choose_engines(now) == ChunkEngines::Both,
        );
        ChunkFitDecision {
            does_chunk_fit_memory,
            has_changed: did_fit_before != does_chunk_fit_memory,
        }
    }

    pub(crate) fn set_paused(&self, is_paused: bool) {
        self.is_paused.set(is_paused);
    }

    /// Programs have fixed shapes, so tails, reduced chunks, and one-token decode
    /// all stay on the GPU. Fixed-shape execution never pads tokens to fit.
    fn handles(&self, token_count: i32) -> bool {
        !self.has_failed.get()
            && !self.is_paused.get()
            && self.does_chunk_use_programs.get()
            && u32::try_from(token_count) == Ok(self.chunk_token_count)
    }

    /// Where the chunk that just ran went. Asked before the comparison hears about it,
    /// because that moves on to the next chunk's turn.
    fn placement_of_finished_chunk(&self, token_count: usize) -> NeuralEngineChunkPlacement {
        if self.has_failed.get() {
            NeuralEngineChunkPlacement::GpuProgramFailed
        } else if u32::try_from(token_count) != Ok(self.chunk_token_count) {
            NeuralEngineChunkPlacement::GpuChunkNotFull
        } else if !self.does_chunk_fit_memory.get() {
            NeuralEngineChunkPlacement::GpuMemoryTight
        } else if self.does_chunk_use_programs.get() {
            NeuralEngineChunkPlacement::BothEngines
        } else if self.chunk_comparison.borrow().did_lead_this_chunk() {
            NeuralEngineChunkPlacement::GpuWakingPrograms
        } else {
            NeuralEngineChunkPlacement::GpuWhileComparing
        }
    }

    fn note_finished_chunk(
        &self,
        start_token: usize,
        token_count: usize,
        elapsed: Duration,
        now: Instant,
    ) -> NeuralEngineChunkReport {
        let placement = self.placement_of_finished_chunk(token_count);
        let mut chunk_comparison = self.chunk_comparison.borrow_mut();
        if self.does_chunk_fit_memory.get() {
            chunk_comparison.note_finished_chunk(start_token, token_count, elapsed, now);
        } else {
            chunk_comparison.note_uncomparable_chunk(start_token, token_count);
        }
        NeuralEngineChunkReport {
            placement,
            chunk_token_count: u32::try_from(token_count).unwrap_or(u32::MAX),
            neural_engine_token_count: if placement == NeuralEngineChunkPlacement::BothEngines {
                self.neural_engine_token_count
            } else {
                0
            },
            elapsed_millis: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            is_prompt_warm: chunk_comparison.is_prompt_warm(),
        }
    }
}

impl Qwen3_5ResidentModel {
    /// The attached programs' surfaces, which a reading of wired growth cannot see.
    #[cfg(feature = "direct-mlx")]
    pub fn neural_engine_program_surface_bytes(&self) -> u64 {
        match self.neural_engine_mlp_prefill.as_ref() {
            Some(prefill) => prefill.surface_bytes,
            None => 0,
        }
    }

    /// Runs every attached program once with the GPU idle, so a measurement can read
    /// what the programs hold with none of MLX's own working set in the reading.
    #[cfg(feature = "direct-mlx")]
    pub fn run_every_neural_engine_program_once(&self) -> Result<(), String> {
        if let Some(prefill) = self.neural_engine_mlp_prefill.as_ref() {
            for program in prefill.layer_programs.borrow_mut().iter_mut() {
                if let Err(program_error) = program.run_once_on_zeros(NEURAL_ENGINE_FINISH_TIMEOUT)
                {
                    return Err(program_error.to_string());
                }
            }
        }
        if let Some(output_gate_engine) = self.base.output_gate_engine.as_ref() {
            output_gate_engine.run_every_program_once()?;
        }
        Ok(())
    }

    /// Decides whether the coming chunk uses the programs, from the peak MLX is
    /// projected to reach and what the comparison wants. Asked per chunk because the
    /// decoder state grows with the prompt, so one prompt can start on both engines and
    /// finish on the GPU.
    pub(crate) fn decide_neural_engine_for_chunk(
        &self,
        chunk_token_count: usize,
        projected_chunk_peak_bytes: u64,
        active_memory_ceiling_bytes: u64,
    ) {
        let Some(prefill) = self.neural_engine_mlp_prefill.as_ref() else {
            return;
        };
        let chunk_fit = prefill.decide_chunk(
            chunk_token_count,
            projected_chunk_peak_bytes,
            active_memory_ceiling_bytes,
            Instant::now(),
        );
        if chunk_fit.has_changed {
            tracing::info!(
                does_chunk_fit_memory = chunk_fit.does_chunk_fit_memory,
                projected_chunk_peak_bytes,
                measured_program_bytes = prefill.measured_program_bytes,
                active_memory_ceiling_bytes,
                "Neural Engine prefill changed engines for the coming chunk"
            );
        }
        if let Some(output_gate_engine) = self.base.output_gate_engine.as_ref() {
            output_gate_engine.set_chunk_uses_programs(prefill.does_chunk_use_programs.get());
        }
    }

    /// Tells the comparison how long a prompt chunk took on the engines chosen for it,
    /// and says where it ran. `None` when this model has no programs attached.
    pub(crate) fn note_neural_engine_chunk_finished(
        &self,
        start_token: usize,
        token_count: usize,
        elapsed: Duration,
    ) -> Option<NeuralEngineChunkReport> {
        let prefill = self.neural_engine_mlp_prefill.as_ref()?;
        Some(prefill.note_finished_chunk(start_token, token_count, elapsed, Instant::now()))
    }
}

/// How many of a chunk's tokens the Neural Engine takes, or `None` when the chunk
/// is too small for a whole multiple of the granularity.
fn neural_engine_tokens_for_chunk(chunk_token_count: u32) -> Option<u32> {
    let share_token_count = u64::from(chunk_token_count) * u64::from(NEURAL_ENGINE_SHARE_NUMERATOR)
        / u64::from(NEURAL_ENGINE_SHARE_DENOMINATOR);
    let aligned_token_count = u32::try_from(share_token_count).ok()?
        / NEURAL_ENGINE_TOKEN_GRANULARITY
        * NEURAL_ENGINE_TOKEN_GRANULARITY;
    (aligned_token_count > 0).then_some(aligned_token_count)
}

#[cfg(test)]
#[path = "neural_engine_mlp_prefill/neural_engine_mlp_prefill_tests.rs"]
mod tests;
