//! Plans one Neural Engine MLP program per decoder layer when a model loads, decides
//! whether they may run at all, and writes the compile input for each.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};
use astronomical_memory_policy::neural_engine_admission::{
    NeuralEnginePrefillAdmission, NeuralEnginePrefillMemoryRequirements,
};
use astronomical_neural_engine::core_ml::CoreMlProgram;
use astronomical_neural_engine::core_ml::compute_device_report::report_compute_devices;
use astronomical_neural_engine::documents::contraction_split::contraction_part_width;
use astronomical_neural_engine::documents::mlp::{MlpFp16Weights, encode_ml_program_mlp};
use astronomical_neural_engine::documents::package_layout::write_ml_package;
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxDtype;

use super::{NeuralEngineMlpPrefill, neural_engine_tokens_for_chunk};
use crate::qwen3_5_core::dense::mlp::Qwen3_5DenseMlpWeights;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_resident::host_description::current_host_environment;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::neural_engine_chunk_comparison::{
    ChunkComparison, ChunkComparisonVerdict, not_worthwhile_reason,
};
use crate::qwen3_5_resident::neural_engine_output_gate_prefill::{
    OUTPUT_GATE_PROGRAM_FORMAT_VERSION, least_output_gate_bytes,
};
use crate::qwen3_5_resident::neural_engine_program_cache::{ProgramSetCache, kept_program_path};
use crate::qwen3_5_resident::neural_engine_program_memory_record::{
    record_program_memory_bytes, recorded_program_memory_bytes,
};
use crate::qwen3_5_resident::neural_engine_program_preparation::{
    NeuralEngineProgramFamily, PlannedProgram, ProgramIdentity,
};
use crate::qwen3_5_resident::neural_engine_program_speed_record::ProgramSpeedRecord;

pub(crate) const FP16_BYTES_PER_WEIGHT: u64 = 2;
const WEIGHT_READ_FAILURE: &str = "a model weight could not be read as fp16";
// The gigabyte the configured ceiling uses.
const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;
// gate + up + down: `[2 · intermediate][hidden]` and `[hidden][intermediate]`.
const INTERMEDIATE_BY_HIDDEN_BLOCKS_PER_LAYER: u64 = 3;
const PROGRAM_FAMILY: &str = "mlp";
// Names the kept programs' set. Raise it whenever the encoded program changes, so
// a load never runs a program compiled from an older encoding.
const PROGRAM_FORMAT_VERSION: u32 = 1;
// Raise it whenever measuring changes, so no load trusts a reading of the old method.
const MEASUREMENT_METHOD_VERSION: u32 = 1;

/// Rounded up so the figure a user reads is one they can set the ceiling to.
const fn whole_gigabytes_rounded_up(byte_count: u64) -> u64 {
    byte_count.div_ceil(BYTES_PER_GIGABYTE)
}

pub(crate) fn unavailable(reason: &str) -> NeuralEnginePrefillStatus {
    tracing::warn!(
        reason,
        "Neural Engine prefill is unavailable; prefill stays on the GPU"
    );
    NeuralEnginePrefillStatus::Unavailable {
        reason: reason.to_owned(),
    }
}

/// The MLP programs a model will get, once it has been decided they may run.
pub(crate) struct MlpProgramPlan {
    shape: ProgramShape,
    intermediate_channels: usize,
    chunk_token_count: u32,
    neural_engine_token_count: u32,
    contraction_part_width: usize,
    weight_bytes_per_program: u64,
    surface_bytes_per_program: u64,
    decoder_layer_count: usize,
    program_cache: Option<ProgramSetCache>,
    speed_record: Option<ProgramSpeedRecord>,
}

/// Everything that can refuse the programs is decided here, before any is built, so
/// the status is settled immediately for a model that will never use them. The error
/// is the status to report. Whatever goes wrong leaves the model serving on the GPU,
/// because an opt-in speedup must never stop a model from loading.
pub(crate) fn plan_neural_engine_mlp_programs(
    model: &Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    program_cache_directory: Option<&Path>,
    neural_engine_token_count_override: Option<u32>,
) -> Result<MlpProgramPlan, NeuralEnginePrefillStatus> {
    if !chunking.experimental_neural_engine_prefill_enabled {
        return Err(NeuralEnginePrefillStatus::Off);
    }
    if !report_compute_devices().has_neural_engine() {
        return Err(unavailable("this Mac reports no Neural Engine"));
    }
    let chunk_token_count = chunking.fixed_prompt_processing_chunk_size_tokens;
    let chosen_token_count =
        neural_engine_token_count_override.or(neural_engine_tokens_for_chunk(chunk_token_count));
    let Some(neural_engine_token_count) = chosen_token_count else {
        return Err(unavailable("the prompt-processing chunk is too small"));
    };
    let Some(layer_weights) = dense_mlp_layer_weights(model) else {
        return Err(unavailable("only dense models are supported"));
    };
    let hidden_channels = model.base.config().hidden_size() as usize;
    let intermediate_channels = model.base.config().dense_intermediate_size() as usize;
    let Some(contraction_part_width) =
        contraction_part_width(&[hidden_channels, intermediate_channels])
    else {
        return Err(unavailable("the model's MLP shape is not supported"));
    };
    let Ok(shape) = ProgramShape::new(
        hidden_channels,
        hidden_channels,
        neural_engine_token_count as usize,
    ) else {
        return Err(unavailable("the model's MLP shape is not supported"));
    };
    let surface_bytes_per_program = program_surface_bytes(shape);
    let least_program_bytes = mlp_weight_bytes_per_layer(model)
        .saturating_add(surface_bytes_per_program)
        .saturating_mul(layer_weights.len() as u64)
        .saturating_add(least_output_gate_bytes(model, chunk_token_count));
    if let Some(refused_status) = refusal_for_program_bytes(
        model,
        chunk_token_count,
        least_program_bytes,
        ProgramCostCertainty::AtLeast,
    ) {
        return Err(refused_status);
    }
    let program_cache = open_program_cache(
        program_cache_directory,
        neural_engine_token_count,
        chunk_token_count,
    );
    let speed_record = program_speed_record_of(program_cache.as_ref());
    if let Some(ChunkComparisonVerdict::NotWorthwhile { gain }) = speed_record
        .as_ref()
        .and_then(ProgramSpeedRecord::recorded_verdict)
    {
        return Err(unavailable(&not_worthwhile_reason(gain)));
    }
    Ok(MlpProgramPlan {
        shape,
        intermediate_channels,
        chunk_token_count,
        neural_engine_token_count,
        contraction_part_width,
        weight_bytes_per_program: mlp_weight_bytes_per_layer(model),
        surface_bytes_per_program,
        decoder_layer_count: layer_weights.len(),
        program_cache,
        speed_record,
    })
}

/// What a refusal knows about the programs' cost, so its figure is one a user can act on.
#[derive(Clone, Copy)]
enum ProgramCostCertainty {
    /// Before they run: only their weights and surfaces are known.
    AtLeast,
    Measured,
}

/// What the programs compete with for the smallest prompt that uses them.
fn memory_requirements_of(
    model: &Qwen3_5ResidentModel,
    chunk_token_count: u32,
    program_bytes: u64,
) -> NeuralEnginePrefillMemoryRequirements {
    let context_state_bytes_per_token = model
        .base
        .decoder_cache_layout()
        .sequence_state_payload_byte_count_per_token()
        .unwrap_or(0);
    NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: model.base.resident_model_payload_byte_count(),
        one_chunk_state_bytes: u64::try_from(context_state_bytes_per_token)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::from(chunk_token_count)),
        program_bytes,
        active_memory_ceiling_bytes: u64::try_from(
            model.runtime().memory_limits().active_memory_limit_bytes(),
        )
        .unwrap_or(u64::MAX),
    }
}

/// The status to report when the programs cannot fit even for one chunk of a prompt.
fn refusal_for_program_bytes(
    model: &Qwen3_5ResidentModel,
    chunk_token_count: u32,
    program_bytes: u64,
    program_cost_certainty: ProgramCostCertainty,
) -> Option<NeuralEnginePrefillStatus> {
    let requirements = memory_requirements_of(model, chunk_token_count, program_bytes);
    match requirements.decide() {
        NeuralEnginePrefillAdmission::Admitted => None,
        NeuralEnginePrefillAdmission::Refused { required_bytes } => Some(unavailable(&format!(
            "the programs need {} {} GB of memory including {} GB for themselves, but the MLX memory ceiling is {} GB; a higher ceiling takes effect when the model next loads",
            match program_cost_certainty {
                ProgramCostCertainty::AtLeast => "at least",
                ProgramCostCertainty::Measured => "about",
            },
            whole_gigabytes_rounded_up(required_bytes),
            whole_gigabytes_rounded_up(program_bytes),
            requirements.active_memory_ceiling_bytes / BYTES_PER_GIGABYTE,
        ))),
    }
}

impl MlpProgramPlan {
    /// Asks again with what the programs measured, at the ceiling in force now: it can
    /// be lowered during the minute or more they take to build.
    pub(crate) fn refusal_for_measured_bytes(
        &self,
        model: &Qwen3_5ResidentModel,
        measured_program_bytes: u64,
    ) -> Option<NeuralEnginePrefillStatus> {
        refusal_for_program_bytes(
            model,
            self.chunk_token_count,
            measured_program_bytes,
            ProgramCostCertainty::Measured,
        )
    }

    /// Whether a cost would be admitted, with no status to report. A record that fails
    /// this is measured again rather than taken at its word.
    pub(crate) fn fits_ceiling(&self, model: &Qwen3_5ResidentModel, program_bytes: u64) -> bool {
        matches!(
            memory_requirements_of(model, self.chunk_token_count, program_bytes).decide(),
            NeuralEnginePrefillAdmission::Admitted
        )
    }

    /// One layer's surfaces, which are wired when a program loads and so before any
    /// reading of what its first runs add.
    pub(crate) const fn surface_bytes_per_program(&self) -> u64 {
        self.surface_bytes_per_program
    }

    /// What one layer's program holds as fp16, the floor of what it can cost.
    pub(crate) const fn weight_bytes_per_program(&self) -> u64 {
        self.weight_bytes_per_program
    }

    /// What this set wired on an earlier load here, when it can be trusted.
    pub(crate) fn recorded_wired_bytes(&self, weight_bytes: u64) -> Option<u64> {
        recorded_program_memory_bytes(
            self.program_cache.as_ref()?.set_directory(),
            &measured_for(current_host_environment()?),
            weight_bytes,
        )
    }

    pub(crate) fn record_wired_bytes(&self, weight_bytes: u64, wired_bytes: u64) {
        if let (Some(program_cache), Some(host_environment)) =
            (self.program_cache.as_ref(), current_host_environment())
        {
            record_program_memory_bytes(
                program_cache.set_directory(),
                &measured_for(host_environment),
                weight_bytes,
                wired_bytes,
            );
        }
    }

    pub(crate) fn planned_programs(&self) -> Vec<PlannedProgram> {
        let mut planned_programs = Vec::with_capacity(self.decoder_layer_count);
        for decoder_layer_index in 0..self.decoder_layer_count {
            planned_programs.push(PlannedProgram {
                identity: ProgramIdentity {
                    family: NeuralEngineProgramFamily::Mlp,
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
            .ok_or(WEIGHT_READ_FAILURE)?;
        let Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
            &decoder_layer_weights.mlp_weights
        else {
            return Err(WEIGHT_READ_FAILURE);
        };
        MlpProgramSource {
            runtime: model.runtime(),
            dense_mlp_weights,
            shape: self.shape,
            intermediate_channels: self.intermediate_channels,
            contraction_part_width: self.contraction_part_width,
        }
        .write_package(package_directory)
    }

    /// Puts the finished programs in the model's path, which takes effect at the next
    /// chunk, and reports the status a reader should now see.
    pub(crate) fn attach(
        &self,
        model: &mut Qwen3_5ResidentModel,
        layer_programs: Vec<CoreMlProgram>,
        measured_program_bytes: u64,
        surface_bytes: u64,
    ) -> NeuralEnginePrefillStatus {
        if let Some(program_cache) = &self.program_cache {
            program_cache.record_finished_set(layer_programs.len() as u64);
        }
        let layer_program_count = u32::try_from(layer_programs.len()).unwrap_or(u32::MAX);
        let recorded_verdict = self
            .speed_record
            .as_ref()
            .and_then(ProgramSpeedRecord::recorded_verdict);
        let full_chunk_token_count = self.chunk_token_count as usize;
        let chunk_comparison = match recorded_verdict {
            Some(verdict) => ChunkComparison::already_decided(full_chunk_token_count, verdict),
            None => ChunkComparison::new(full_chunk_token_count),
        };
        model.neural_engine_mlp_prefill = Some(NeuralEngineMlpPrefill {
            layer_programs: RefCell::new(layer_programs),
            chunk_token_count: self.chunk_token_count,
            neural_engine_token_count: self.neural_engine_token_count,
            measured_program_bytes,
            surface_bytes,
            has_failed: Cell::new(false),
            is_paused: Cell::new(false),
            does_chunk_fit_memory: Cell::new(true),
            does_chunk_use_programs: Cell::new(true),
            chunk_comparison: RefCell::new(chunk_comparison),
            speed_record: self.speed_record.clone(),
            has_reported_verdict: Cell::new(recorded_verdict.is_some()),
        });
        NeuralEnginePrefillStatus::Active {
            layer_program_count,
            neural_engine_tokens_per_chunk: self.neural_engine_token_count,
            chunk_token_count: self.chunk_token_count,
            measured_memory_bytes: measured_program_bytes,
            measured_speed_percent: recorded_verdict.map(ChunkComparisonVerdict::speed_percent),
        }
    }
}

fn program_speed_record_of(program_cache: Option<&ProgramSetCache>) -> Option<ProgramSpeedRecord> {
    Some(ProgramSpeedRecord::new(
        program_cache?.set_directory(),
        measured_for(current_host_environment()?),
    ))
}

/// The set is named by everything a compiled program's contents depend on besides
/// the model revision, which names the folder above it.
fn open_program_cache(
    program_cache_directory: Option<&Path>,
    neural_engine_token_count: u32,
    chunk_token_count: u32,
) -> Option<ProgramSetCache> {
    ProgramSetCache::open(
        program_cache_directory?,
        PROGRAM_FAMILY,
        &format!("v{PROGRAM_FORMAT_VERSION}-{neural_engine_token_count}-of-{chunk_token_count}"),
    )
}

fn dense_mlp_layer_weights(model: &Qwen3_5ResidentModel) -> Option<Vec<&Qwen3_5DenseMlpWeights>> {
    let mut layer_weights = Vec::with_capacity(model.base.weights.decoder_layer_weights.len());
    for decoder_layer_weights in &model.base.weights.decoder_layer_weights {
        match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => {
                layer_weights.push(dense_mlp_weights);
            }
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(_) => return None,
        }
    }
    (!layer_weights.is_empty()).then_some(layer_weights)
}

/// Everything a kept reading depends on besides the set's folder. The output gates are
/// measured with the MLP programs but keep their own format version.
fn measured_for(host_environment: &str) -> String {
    format!(
        "{host_environment}; measurement method {MEASUREMENT_METHOD_VERSION}; output gates {OUTPUT_GATE_PROGRAM_FORMAT_VERSION}"
    )
}

/// One program's input and output surfaces, as fp16.
pub(crate) fn program_surface_bytes(shape: ProgramShape) -> u64 {
    ((shape.activation_element_count() + shape.projected_element_count()) as u64)
        .saturating_mul(FP16_BYTES_PER_WEIGHT)
}

/// What one layer's whole MLP holds as fp16.
fn mlp_weight_bytes_per_layer(model: &Qwen3_5ResidentModel) -> u64 {
    let config = model.base.config();
    INTERMEDIATE_BY_HIDDEN_BLOCKS_PER_LAYER
        .saturating_mul(u64::from(config.dense_intermediate_size()))
        .saturating_mul(u64::from(config.hidden_size()))
        .saturating_mul(FP16_BYTES_PER_WEIGHT)
}

/// The weight as fp16 bit patterns, row-major `[output][input]`. Dequantizing to
/// fp16 is why this mode is opt-in: it rounds the artifact's values to within
/// 0.3 percent of the weight RMS.
pub(crate) fn weight_as_fp16_bits(
    runtime: &MlxRuntime,
    affine_weights: &Qwen3_5AffineWeights,
) -> Result<Vec<u16>, &'static str> {
    let dequantized = match affine_weights {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => runtime.copy(weight),
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_group_size,
            quantization_bits,
        } => runtime.dequantize_affine(
            packed_weight,
            quantization_scales,
            quantization_biases,
            *quantization_group_size,
            *quantization_bits,
        ),
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => {
            return Err(WEIGHT_READ_FAILURE);
        }
    };
    let dequantized = dequantized.map_err(read_failure)?;
    let half_precision = runtime
        .astype(&dequantized, MlxDtype::Float16)
        .map_err(read_failure)?;
    runtime
        .evaluate_arrays(&[&half_precision])
        .map_err(read_failure)?;
    half_precision.to_vec_f16_bits().map_err(read_failure)
}

fn read_failure(_mlx_error: mlx_bridge::MlxCError) -> &'static str {
    WEIGHT_READ_FAILURE
}

/// One decoder layer's whole MLP, read from the model as fp16 when it is compiled.
struct MlpProgramSource<'model> {
    runtime: &'model MlxRuntime,
    dense_mlp_weights: &'model Qwen3_5DenseMlpWeights,
    shape: ProgramShape,
    intermediate_channels: usize,
    contraction_part_width: usize,
}

impl MlpProgramSource<'_> {
    /// `(gate then up, down)`, the two blocks the encoder takes.
    fn fp16_weights(&self) -> Result<(Vec<u16>, Vec<u16>), &'static str> {
        let mut gate_and_up =
            weight_as_fp16_bits(self.runtime, &self.dense_mlp_weights.gate_projection)?;
        gate_and_up.extend(weight_as_fp16_bits(
            self.runtime,
            &self.dense_mlp_weights.up_projection,
        )?);
        let down = weight_as_fp16_bits(self.runtime, &self.dense_mlp_weights.down_projection)?;
        Ok((gate_and_up, down))
    }

    fn write_package(&self, package_directory: &Path) -> Result<PathBuf, &'static str> {
        let (gate_and_up, down) = self.fp16_weights()?;
        let Ok(package_contents) = encode_ml_program_mlp(
            self.shape,
            self.intermediate_channels,
            MlpFp16Weights {
                gate_and_up: &gate_and_up,
                down: &down,
            },
            self.contraction_part_width,
        ) else {
            return Err("the model's MLP weights do not fit a Neural Engine program");
        };
        match write_ml_package(package_directory, &package_contents) {
            Ok(package_path) => Ok(package_path),
            Err(_) => Err("could not write the MLP program package"),
        }
    }
}

#[cfg(test)]
#[path = "program_loading_tests.rs"]
mod tests;
