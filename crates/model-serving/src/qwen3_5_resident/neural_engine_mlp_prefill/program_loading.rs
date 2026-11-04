//! Builds one Neural Engine MLP program per decoder layer when a model loads,
//! and decides whether they may run at all.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::time::Instant;

use astronomical_ipc_protocol::{NeuralEnginePrefillStatus, WorkerChunkingConfiguration};
use astronomical_memory_policy::neural_engine_admission::{
    NeuralEnginePrefillAdmission, NeuralEnginePrefillMemoryRequirements,
};
use astronomical_neural_engine::compute_device_report::report_compute_devices;
use astronomical_neural_engine::core_ml_projection::{
    ComputePlacement, CoreMlComputeUnits, CoreMlProjection,
};
use astronomical_neural_engine::ml_program_mlp_document::MlpFp16Weights;
use astronomical_neural_engine::projection_shape::ProjectionShape;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxDtype;

use super::{NeuralEngineMlpPrefill, neural_engine_tokens_for_chunk};
use crate::qwen3_5_core::dense::mlp::Qwen3_5DenseMlpWeights;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

// A 2,048-deep contraction runs at 18.6 TFLOP/s on the Neural Engine and a
// 4,096-deep one at 9.7 (shape sweep, issue #161).
const CONTRACTION_PART_WIDTH: u32 = 2048;
const FP16_BYTES_PER_WEIGHT: u64 = 2;
const WEIGHT_READ_FAILURE: &str = "a model weight could not be read as fp16";
// The gigabyte the configured ceiling uses.
const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;
// gate + up + down: `[2 · intermediate][hidden]` and `[hidden][intermediate]`.
const INTERMEDIATE_BY_HIDDEN_BLOCKS_PER_LAYER: u64 = 3;

/// Rounded up so the figure a user reads is one they can set the ceiling to.
const fn whole_gigabytes_rounded_up(byte_count: u64) -> u64 {
    byte_count.div_ceil(BYTES_PER_GIGABYTE)
}

fn unavailable(reason: &str) -> NeuralEnginePrefillStatus {
    tracing::warn!(
        reason,
        "Neural Engine MLP prefill is unavailable; prefill stays on the GPU"
    );
    NeuralEnginePrefillStatus::Unavailable {
        reason: reason.to_owned(),
    }
}

/// Whatever goes wrong here leaves the model serving on the GPU, because an
/// opt-in speedup must never stop a model from loading.
pub(crate) fn prepare_neural_engine_mlp_prefill(
    model: &mut Qwen3_5ResidentModel,
    chunking: &WorkerChunkingConfiguration,
    maximum_context_tokens: u32,
) -> NeuralEnginePrefillStatus {
    if !chunking.experimental_neural_engine_mlp_prefill_enabled {
        return NeuralEnginePrefillStatus::Off;
    }
    if !report_compute_devices().has_neural_engine() {
        return unavailable("this Mac reports no Neural Engine");
    }
    let chunk_token_count = chunking.fixed_prompt_processing_chunk_size_tokens;
    let Some(neural_engine_token_count) = neural_engine_tokens_for_chunk(chunk_token_count) else {
        return unavailable("the prompt-processing chunk is too small");
    };
    let Some(layer_weights) = dense_mlp_layer_weights(model) else {
        return unavailable("only dense models are supported");
    };
    let estimated_memory_bytes = match admit_programs(model, maximum_context_tokens) {
        NeuralEnginePrefillAdmission::Admitted {
            estimated_neural_engine_bytes,
        } => estimated_neural_engine_bytes,
        NeuralEnginePrefillAdmission::Refused {
            estimated_neural_engine_bytes,
            required_bytes,
            active_memory_ceiling_bytes,
        } => {
            return unavailable(&format!(
                "the programs need about {} GB of memory including {} GB for themselves, but the MLX memory ceiling is {} GB",
                whole_gigabytes_rounded_up(required_bytes),
                whole_gigabytes_rounded_up(estimated_neural_engine_bytes),
                active_memory_ceiling_bytes / BYTES_PER_GIGABYTE,
            ));
        }
    };
    let layer_programs = match load_layer_programs(model, &layer_weights, neural_engine_token_count)
    {
        Ok(layer_programs) => layer_programs,
        Err(reason) => return unavailable(reason),
    };
    let layer_program_count = u32::try_from(layer_programs.len()).unwrap_or(u32::MAX);
    model.neural_engine_mlp_prefill = Some(NeuralEngineMlpPrefill {
        layer_programs: RefCell::new(layer_programs),
        chunk_token_count,
        neural_engine_token_count,
        has_failed: Cell::new(false),
    });
    NeuralEnginePrefillStatus::Active {
        layer_program_count,
        neural_engine_tokens_per_chunk: neural_engine_token_count,
        chunk_token_count,
        estimated_memory_bytes,
    }
}

/// Each layer's MLP weights, or `None` for any non-dense layer.
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

fn admit_programs(
    model: &Qwen3_5ResidentModel,
    maximum_context_tokens: u32,
) -> NeuralEnginePrefillAdmission {
    let config = model.base.config();
    let context_state_bytes_per_token = model
        .base
        .decoder_cache_layout()
        .sequence_state_payload_byte_count_per_token()
        .unwrap_or(0);
    NeuralEnginePrefillMemoryRequirements {
        resident_model_payload_bytes: model.base.resident_model_payload_byte_count(),
        maximum_context_state_bytes: u64::try_from(context_state_bytes_per_token)
            .unwrap_or(u64::MAX)
            .saturating_mul(u64::from(maximum_context_tokens)),
        fp16_mlp_weight_bytes: u64::from(config.layer_count())
            .saturating_mul(INTERMEDIATE_BY_HIDDEN_BLOCKS_PER_LAYER)
            .saturating_mul(u64::from(config.dense_intermediate_size()))
            .saturating_mul(u64::from(config.hidden_size()))
            .saturating_mul(FP16_BYTES_PER_WEIGHT),
        active_memory_ceiling_bytes: u64::try_from(
            model.runtime().memory_limits().active_memory_limit_bytes(),
        )
        .unwrap_or(u64::MAX),
    }
    .decide()
}

/// The weight as fp16 bit patterns, row-major `[output][input]`. Dequantizing to
/// fp16 is why this mode is opt-in: it rounds the artifact's values to within
/// 0.3 percent of the weight RMS.
fn weight_as_fp16_bits(
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

fn read_failure(_mlx_error: mlx_c_rust::MlxCError) -> &'static str {
    WEIGHT_READ_FAILURE
}

/// Compile input for one program, removed as soon as Core ML holds its own compiled
/// copy, because each one is nearly a whole layer's weights on disk.
struct ProgramPackageDirectory {
    path: PathBuf,
}

impl ProgramPackageDirectory {
    fn create(layer_index: usize) -> Option<Self> {
        let path = std::env::temp_dir().join(format!(
            "astronomical-neural-engine-{}-{layer_index}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).ok()?;
        Some(Self { path })
    }
}

impl Drop for ProgramPackageDirectory {
    fn drop(&mut self) {
        if let Err(removal_error) = std::fs::remove_dir_all(&self.path) {
            tracing::warn!(error = %removal_error, "could not remove a Neural Engine program package");
        }
    }
}

fn load_layer_program(
    model: &Qwen3_5ResidentModel,
    dense_mlp_weights: &Qwen3_5DenseMlpWeights,
    layer_index: usize,
    neural_engine_token_count: u32,
) -> Result<CoreMlProjection, &'static str> {
    let runtime = model.runtime();
    let hidden_channels = model.base.config().hidden_size() as usize;
    let mut gate_and_up = weight_as_fp16_bits(runtime, &dense_mlp_weights.gate_projection)?;
    gate_and_up.extend(weight_as_fp16_bits(
        runtime,
        &dense_mlp_weights.up_projection,
    )?);
    let down = weight_as_fp16_bits(runtime, &dense_mlp_weights.down_projection)?;
    let Ok(shape) = ProjectionShape::new(
        hidden_channels,
        hidden_channels,
        neural_engine_token_count as usize,
    ) else {
        return Err("the model's MLP shape is not supported");
    };
    let Some(package_directory) = ProgramPackageDirectory::create(layer_index) else {
        return Err("no scratch directory for the Core ML program");
    };
    let Ok(mut program) = CoreMlProjection::load_ml_program_mlp(
        shape,
        model.base.config().dense_intermediate_size() as usize,
        MlpFp16Weights {
            gate_and_up: &gate_and_up,
            down: &down,
        },
        CONTRACTION_PART_WIDTH as usize,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        &package_directory.path,
    ) else {
        return Err("Core ML could not load the MLP program on this macOS");
    };
    let placement = program.placement();
    if placement.preferred != ComputePlacement::NeuralEngine
        || placement.operations_off_neural_engine != 0
    {
        return Err("Core ML did not place the whole MLP on the Neural Engine");
    }
    // A program's first prediction wires its weights and is far slower than the
    // rest, so running it here keeps that cost out of a user's first long prompt,
    // and a program that cannot run fails the load instead of a request.
    if program.predict().is_err() {
        return Err("the Neural Engine could not run the MLP program");
    }
    Ok(program)
}

fn load_layer_programs(
    model: &Qwen3_5ResidentModel,
    layer_weights: &[&Qwen3_5DenseMlpWeights],
    neural_engine_token_count: u32,
) -> Result<Vec<CoreMlProjection>, &'static str> {
    let started_at = Instant::now();
    let mut layer_programs = Vec::with_capacity(layer_weights.len());
    for (layer_index, dense_mlp_weights) in layer_weights.iter().enumerate() {
        layer_programs.push(load_layer_program(
            model,
            dense_mlp_weights,
            layer_index,
            neural_engine_token_count,
        )?);
        // Loading takes over a minute on a large model, so the wait stays visible.
        let completed_layers = layer_index + 1;
        let elapsed_seconds = started_at.elapsed().as_secs_f64();
        let remaining_seconds = elapsed_seconds / completed_layers as f64
            * (layer_weights.len() - completed_layers) as f64;
        tracing::info!(
            completed_layers,
            total_layers = layer_weights.len(),
            percent_complete = 100 * completed_layers / layer_weights.len(),
            elapsed_seconds,
            eta_seconds = remaining_seconds,
            "Neural Engine MLP programs loading"
        );
    }
    if model
        .runtime()
        .synchronize_gpu_stream_and_clear_allocator_cache()
        .is_err()
    {
        return Err("the MLX allocator could not settle after reading the weights");
    }
    Ok(layer_programs)
}
