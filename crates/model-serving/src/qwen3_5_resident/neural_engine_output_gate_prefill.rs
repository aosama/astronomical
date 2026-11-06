//! Linear-attention output gates on the Neural Engine during dense prefill.
//!
//! Each DeltaNet layer's gate is `4096 → 4096` at every token of a chunk, with the
//! other projections on the GPU at the same moment. See
//! [`LinearAttentionOutputGateEngine`] for why only these rows are offered.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::time::Duration;

use astronomical_ipc_protocol::WorkerChunkingConfiguration;
use astronomical_neural_engine::core_ml::CoreMlProgram;
use astronomical_neural_engine::documents::package_layout::write_ml_package;
use astronomical_neural_engine::documents::projection::{
    MlProgramWeights, encode_ml_program_projection,
};
use astronomical_neural_engine::mlx_hand_off::{
    OutputHandOff, channel_major_fp16, surface_as_token_major, write_prepared_input,
};
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use super::model::Qwen3_5ResidentModel;
use super::neural_engine_mlp_prefill::{NEURAL_ENGINE_TOKEN_GRANULARITY, weight_as_fp16_bits};
use super::neural_engine_program_cache::{ProgramSetCache, kept_program_path};
use super::neural_engine_program_preparation::{
    NeuralEngineProgramFamily, PlannedProgram, ProgramIdentity,
};
use crate::qwen3_5_core::model::output_gate_engine::LinearAttentionOutputGateEngine;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights,
};

// Two 2,048-deep parts: the depth the contraction-split journeys measured at the
// Neural Engine's full rate, for a 4,096-wide input.
const CONTRACTION_PART_COUNT: usize = 2;
// A prediction takes milliseconds, so this only bounds a wedged Neural Engine.
const NEURAL_ENGINE_FINISH_TIMEOUT: Duration = Duration::from_secs(30);
const FP16_BYTES_PER_WEIGHT: u64 = 2;
const PROGRAM_FAMILY: &str = "output-gate";
// Names the kept programs' set. Raise it whenever the encoded program changes, so
// a load never runs a program compiled from an older encoding.
const PROGRAM_FORMAT_VERSION: u32 = 1;

pub(crate) struct NeuralEngineOutputGatePrefill {
    /// Indexed by decoder layer; full-attention layers have no gate program.
    layer_programs: RefCell<Vec<Option<CoreMlProgram>>>,
    chunk_token_count: u32,
    has_failed: Cell<bool>,
    is_paused: Cell<bool>,
}

impl std::fmt::Debug for NeuralEngineOutputGatePrefill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NeuralEngineOutputGatePrefill")
            .field("chunk_token_count", &self.chunk_token_count)
            .finish_non_exhaustive()
    }
}

impl NeuralEngineOutputGatePrefill {
    fn run_program(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        decoder_layer_index: usize,
        gpu_work: &[&MlxArray],
    ) -> Result<MlxArray, Box<dyn std::error::Error>> {
        let prepared = channel_major_fp16(runtime, hidden_states)?;
        runtime.evaluate_arrays(&[&prepared])?;
        // The GPU starts first so the host copy below does not leave it idle.
        runtime.async_eval_arrays(gpu_work)?;
        let mut layer_programs = self.layer_programs.borrow_mut();
        let program = layer_programs
            .get_mut(decoder_layer_index)
            .and_then(Option::as_mut)
            .ok_or("no output gate program exists for this decoder layer")?;
        write_prepared_input(&prepared, program)?;
        let pending = program.begin_prediction()?;
        runtime.evaluate_arrays(gpu_work)?;
        pending.finish(NEURAL_ENGINE_FINISH_TIMEOUT)?;
        Ok(surface_as_token_major(
            runtime,
            program,
            gpu_work[0].dtype(),
            OutputHandOff::ZeroCopySurface,
        )?)
    }
}

impl LinearAttentionOutputGateEngine for NeuralEngineOutputGatePrefill {
    fn handles(&self, token_count: i32) -> bool {
        !self.has_failed.get()
            && !self.is_paused.get()
            && u32::try_from(token_count) == Ok(self.chunk_token_count)
    }

    fn project_beside_gpu_work(
        &self,
        runtime: &MlxRuntime,
        hidden_states: &MlxArray,
        decoder_layer_index: usize,
        gpu_work: &[&MlxArray],
    ) -> Result<MlxArray, String> {
        match self.run_program(runtime, hidden_states, decoder_layer_index, gpu_work) {
            Ok(output_gate) => Ok(output_gate),
            Err(program_error) => {
                self.has_failed.set(true);
                Err(program_error.to_string())
            }
        }
    }

    fn set_paused(&self, is_paused: bool) {
        self.is_paused.set(is_paused);
    }
}

/// These programs take every token of a chunk, and MLX can read a surface as one
/// array only when its rows fill their alignment. Refusing here keeps a chunk that
/// cannot work from loading programs that would only fail on the first prompt.
fn chunk_fits_surface_rows(chunk_token_count: u32) -> bool {
    chunk_token_count > 0 && chunk_token_count.is_multiple_of(NEURAL_ENGINE_TOKEN_GRANULARITY)
}

/// One linear-attention layer's output gate projection as its own program.
struct OutputGateProgramSource<'model> {
    runtime: &'model MlxRuntime,
    output_gate_projection: &'model Qwen3_5AffineWeights,
    shape: ProgramShape,
}

impl OutputGateProgramSource<'_> {
    fn write_package(&self, package_directory: &Path) -> Result<PathBuf, &'static str> {
        let fp16_weights = weight_as_fp16_bits(self.runtime, self.output_gate_projection)?;
        let Ok(package_contents) = encode_ml_program_projection(
            self.shape,
            MlProgramWeights::Fp16(&fp16_weights),
            CONTRACTION_PART_COUNT,
        ) else {
            return Err("the model's output gate weights do not fit a Neural Engine program");
        };
        match write_ml_package(package_directory, &package_contents) {
            Ok(package_path) => Ok(package_path),
            Err(_) => Err("could not write the output gate program package"),
        }
    }
}

fn linear_attention_layer_count(model: &Qwen3_5ResidentModel) -> u64 {
    let mut layer_count = 0;
    for decoder_layer_weights in &model.base.weights.decoder_layer_weights {
        if matches!(
            decoder_layer_weights.attention_weights,
            Qwen3_5AttentionWeights::Linear(_)
        ) {
            layer_count += 1;
        }
    }
    layer_count
}

/// What every output gate program holds as fp16, for the memory admission that
/// decides whether any program may load.
pub(crate) fn output_gate_fp16_weight_bytes(model: &Qwen3_5ResidentModel) -> u64 {
    let config = model.base.config();
    linear_attention_layer_count(model)
        .saturating_mul(u64::from(config.linear_value_dimension()))
        .saturating_mul(u64::from(config.hidden_size()))
        .saturating_mul(FP16_BYTES_PER_WEIGHT)
}

/// Other layers have no output gate and get no program.
fn linear_attention_layer_indices(model: &Qwen3_5ResidentModel) -> Vec<usize> {
    let mut layer_indices = Vec::new();
    let decoder_layer_weights = &model.base.weights.decoder_layer_weights;
    for (decoder_layer_index, layer_weights) in decoder_layer_weights.iter().enumerate() {
        if matches!(
            layer_weights.attention_weights,
            Qwen3_5AttentionWeights::Linear(_)
        ) {
            layer_indices.push(decoder_layer_index);
        }
    }
    layer_indices
}

/// The output gate programs a model will get. Made only after the MLP programs were
/// admitted, which already counted these.
pub(crate) struct OutputGateProgramPlan {
    shape: ProgramShape,
    chunk_token_count: u32,
    decoder_layer_count: usize,
    linear_attention_layer_indices: Vec<usize>,
    program_cache: Option<ProgramSetCache>,
}

pub(crate) fn plan_neural_engine_output_gate_programs(
    model: &Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    program_cache_directory: Option<&Path>,
) -> Result<OutputGateProgramPlan, &'static str> {
    let chunk_token_count = chunking.fixed_prompt_processing_chunk_size_tokens;
    if !chunk_fits_surface_rows(chunk_token_count) {
        return Err("the prompt-processing chunk is not a multiple of the surface alignment");
    }
    let config = model.base.config();
    let Ok(shape) = ProgramShape::new(
        config.hidden_size() as usize,
        config.linear_value_dimension() as usize,
        chunk_token_count as usize,
    ) else {
        return Err("the output gate shape is not supported");
    };
    let linear_attention_layer_indices = linear_attention_layer_indices(model);
    if linear_attention_layer_indices.is_empty() {
        return Err("the model has no linear-attention layers");
    }
    let mut program_cache = None;
    if let Some(program_cache_directory) = program_cache_directory {
        program_cache = ProgramSetCache::open(
            program_cache_directory,
            PROGRAM_FAMILY,
            &format!("v{PROGRAM_FORMAT_VERSION}-{chunk_token_count}"),
        );
    }
    Ok(OutputGateProgramPlan {
        shape,
        chunk_token_count,
        decoder_layer_count: model.base.weights.decoder_layer_weights.len(),
        linear_attention_layer_indices,
        program_cache,
    })
}

impl OutputGateProgramPlan {
    pub(crate) fn planned_programs(&self) -> Vec<PlannedProgram> {
        let mut planned_programs = Vec::with_capacity(self.linear_attention_layer_indices.len());
        for &decoder_layer_index in &self.linear_attention_layer_indices {
            planned_programs.push(PlannedProgram {
                identity: ProgramIdentity {
                    family: NeuralEngineProgramFamily::OutputGate,
                    decoder_layer_index,
                },
                shape: self.shape,
                kept_program_path: kept_program_path(
                    self.program_cache.as_ref(),
                    decoder_layer_index,
                ),
            });
        }
        planned_programs
    }

    /// One layer's compile input, read from the model as fp16.
    pub(crate) fn write_package(
        &self,
        model: &Qwen3_5ResidentModel,
        decoder_layer_index: usize,
        package_directory: &Path,
    ) -> Result<PathBuf, &'static str> {
        let decoder_layer_weights = model
            .base
            .weights
            .decoder_layer_weights
            .get(decoder_layer_index)
            .ok_or("the model has no such decoder layer")?;
        let Qwen3_5AttentionWeights::Linear(linear_attention_weights) =
            &decoder_layer_weights.attention_weights
        else {
            return Err("the decoder layer has no output gate");
        };
        OutputGateProgramSource {
            runtime: model.runtime(),
            output_gate_projection: &linear_attention_weights.output_gate_projection,
            shape: self.shape,
        }
        .write_package(package_directory)
    }

    /// Puts the finished programs in the model's path and returns how many there are.
    pub(crate) fn attach(
        self,
        model: &mut Qwen3_5ResidentModel,
        built_programs: Vec<(usize, CoreMlProgram)>,
    ) -> u32 {
        let layer_program_count = u32::try_from(built_programs.len()).unwrap_or(u32::MAX);
        if let Some(program_cache) = &self.program_cache {
            program_cache.record_finished_set(u64::from(layer_program_count));
        }
        let mut layer_programs = Vec::with_capacity(self.decoder_layer_count);
        layer_programs.resize_with(self.decoder_layer_count, none_program);
        for (decoder_layer_index, program) in built_programs {
            layer_programs[decoder_layer_index] = Some(program);
        }
        model.base.output_gate_engine = Some(Box::new(NeuralEngineOutputGatePrefill {
            layer_programs: RefCell::new(layer_programs),
            chunk_token_count: self.chunk_token_count,
            has_failed: Cell::new(false),
            is_paused: Cell::new(false),
        }));
        layer_program_count
    }
}

const fn none_program() -> Option<CoreMlProgram> {
    None
}

#[cfg(test)]
#[path = "neural_engine_output_gate_prefill_tests.rs"]
mod tests;
