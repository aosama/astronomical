//! Where the shared Qwen 3.5 core parts from the folded engine on a folded
//! ternary pack, layer by layer (issue #2595): the same 16 prompt tokens'
//! embedding rows and every decoder layer's output rows on both, compared by
//! the largest difference. Reading the first layer that differs points at
//! the stage to fix, which a whole-reply comparison cannot.

use mlx_bridge::MlxArray;
use model_serving::prism_hadamard_qwen35::model::PrismHadamardGenerationState;
use serial_test::serial;

use super::shared_core_parity::loaded_shared_core;
use super::support::{
    WARMUP_INPUT_TOKENS, loaded_journey_model, pack_directory, pack_tokenizer,
    romeo_and_juliet_prompt_token_ids,
};

/// Prompt rows traced through every layer.
const TRACED_TOKEN_COUNT: usize = 16;
const DECODER_LAYER_COUNT: i32 = 64;

/// Host copies of one engine's rows: the embedding, then each layer's output.
struct LayerTrace {
    embedding_rows: Vec<f32>,
    layer_output_rows: Vec<Vec<f32>>,
}

#[test]
#[serial]
#[ignore = "loads the real folded pack on both engines on the graphics processor and prints where they part"]
fn should_trace_where_the_shared_core_parts_from_the_folded_engine_by_layer() {
    let tokenizer = pack_tokenizer();
    let prompt_token_ids: Vec<i32> =
        romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS)
            .into_iter()
            .take(TRACED_TOKEN_COUNT)
            .collect();
    let folded_engine_trace = folded_engine_layer_trace(&prompt_token_ids);
    let shared_core_trace = shared_core_layer_trace(&prompt_token_ids);
    eprintln!(
        "[prism-shared-core-by-layer] embedding largest_difference={:.6} folded_scale={:.4}",
        largest_difference(
            &folded_engine_trace.embedding_rows,
            &shared_core_trace.embedding_rows
        ),
        largest_magnitude(&folded_engine_trace.embedding_rows),
    );
    for (layer_index, (folded_rows, core_rows)) in folded_engine_trace
        .layer_output_rows
        .iter()
        .zip(&shared_core_trace.layer_output_rows)
        .enumerate()
    {
        eprintln!(
            "[prism-shared-core-by-layer] layer={layer_index} largest_difference={:.6} folded_scale={:.4} core_scale={:.4}",
            largest_difference(folded_rows, core_rows),
            largest_magnitude(folded_rows),
            largest_magnitude(core_rows),
        );
    }
}

fn folded_engine_layer_trace(prompt_token_ids: &[i32]) -> LayerTrace {
    let folded_engine = loaded_journey_model();
    let mut hidden_rows = folded_engine
        .probe_embedding(prompt_token_ids)
        .expect("the folded engine embeds the prompt");
    let embedding_rows = host_rows(&folded_engine.runtime, &hidden_rows);
    let mut state = PrismHadamardGenerationState::empty();
    let mut layer_output_rows = Vec::with_capacity(DECODER_LAYER_COUNT as usize);
    for layer_index in 0..DECODER_LAYER_COUNT as usize {
        hidden_rows = folded_engine
            .probe_decoder_layer(layer_index, &hidden_rows, &mut state)
            .expect("the folded engine runs the layer");
        layer_output_rows.push(host_rows(&folded_engine.runtime, &hidden_rows));
    }
    LayerTrace {
        embedding_rows,
        layer_output_rows,
    }
}

fn shared_core_layer_trace(prompt_token_ids: &[i32]) -> LayerTrace {
    let shared_core = loaded_shared_core(&pack_directory());
    let runtime = shared_core.runtime();
    let token_indices = runtime
        .array_from_i32(prompt_token_ids, &[1, prompt_token_ids.len() as i32])
        .expect("the token indices upload");
    let embedding_rows = host_rows(
        runtime,
        &shared_core
            .embedding_lookup(&token_indices)
            .expect("the shared core embeds the prompt"),
    );
    let tap_layer_ids: Vec<i32> = (0..DECODER_LAYER_COUNT).collect();
    shared_core.begin_draft_tap_capture(&tap_layer_ids, prompt_token_ids.len());
    let mut decoder_state = super::shared_core_parity::empty_shared_core_state(&shared_core);
    let prompt_token_ids: Vec<u32> = prompt_token_ids
        .iter()
        .map(|token_id| *token_id as u32)
        .collect();
    shared_core
        .forward_chunk_with_pre_final_normalization_hidden_states(
            &prompt_token_ids,
            0,
            &mut decoder_state,
        )
        .expect("the shared core runs the prompt");
    let (_, layer_taps) = shared_core
        .take_draft_taps()
        .expect("the taps join")
        .expect("every layer's rows were kept");
    LayerTrace {
        embedding_rows,
        layer_output_rows: layer_taps
            .iter()
            .map(|layer_tap| host_rows(runtime, layer_tap))
            .collect(),
    }
}

fn host_rows(runtime: &runtime_integration::MlxRuntime, rows: &MlxArray) -> Vec<f32> {
    let float32_rows = runtime
        .astype(rows, mlx_bridge::MlxDtype::Float32)
        .expect("the rows widen");
    runtime
        .array_to_vec_f32(&float32_rows)
        .expect("the rows read back")
}

fn largest_difference(first_rows: &[f32], second_rows: &[f32]) -> f32 {
    if first_rows.len() != second_rows.len() {
        return f32::INFINITY;
    }
    first_rows
        .iter()
        .zip(second_rows)
        .map(|(first_value, second_value)| (first_value - second_value).abs())
        .fold(0.0, f32::max)
}

fn largest_magnitude(rows: &[f32]) -> f32 {
    rows.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

thread_local! {
    /// The folded engine's layer 0 attention rows, read before it is let go.
    static FOLDED_ATTENTION_ROWS: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The folded engine's MLP intermediate rows, read before it is let go.
    static FOLDED_MLP_INTERMEDIATE_ROWS: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The folded pack's RMS norm epsilon.
const RMS_NORM_EPSILON: f32 = 1e-6;
/// Layer 0's fused query, key and value width, before the output gate.
const QUERY_KEY_VALUE_WIDTH: usize = 10_240;

/// Layer 0's input norm and its query, key and value projection on both
/// engines, from the same embedding rows: whether the rotated projection
/// itself or the DeltaNet math after it parts the two.
#[test]
#[serial]
#[ignore = "loads the real folded pack on both engines on the graphics processor and prints where layer 0 parts"]
fn should_trace_the_first_layers_input_projection_on_both_engines() {
    let tokenizer = pack_tokenizer();
    let prompt_token_ids: Vec<i32> =
        romeo_and_juliet_prompt_token_ids(&tokenizer, WARMUP_INPUT_TOKENS)
            .into_iter()
            .take(TRACED_TOKEN_COUNT)
            .collect();
    let (folded_normalized_rows, folded_projected_rows) = {
        let folded_engine = loaded_journey_model();
        let embedding_rows = folded_engine
            .probe_embedding(&prompt_token_ids)
            .expect("the folded engine embeds the prompt");
        let normalized_rows = folded_engine
            .runtime
            .rms_norm(
                &embedding_rows,
                &folded_engine.weights.input_normalization_weights[0],
                RMS_NORM_EPSILON,
            )
            .expect("the folded engine normalizes");
        let folded_mlp_intermediate_rows = folded_engine
            .probe_dense_mlp_intermediate(&normalized_rows)
            .expect("the folded engine runs the MLP's first half");
        FOLDED_MLP_INTERMEDIATE_ROWS.with(|stored_rows| {
            *stored_rows.borrow_mut() =
                host_rows(&folded_engine.runtime, &folded_mlp_intermediate_rows);
        });
        let mut attention_state = PrismHadamardGenerationState::empty();
        let folded_attention_rows = folded_engine
            .probe_linear_attention(0, &normalized_rows, &mut attention_state)
            .expect("the folded engine runs layer 0's attention");
        FOLDED_ATTENTION_ROWS.with(|stored_rows| {
            *stored_rows.borrow_mut() = host_rows(&folded_engine.runtime, &folded_attention_rows);
        });
        let projected_rows = folded_engine
            .probe_packed_projection(
                model_serving::prism_hadamard_qwen35::model::PrismHadamardProbeProjection::GatedDeltaQueryKeyValueAndGate,
                &normalized_rows,
            )
            .expect("the folded engine projects");
        (
            host_rows(&folded_engine.runtime, &normalized_rows),
            host_rows(&folded_engine.runtime, &projected_rows),
        )
    };
    let shared_core = loaded_shared_core(&pack_directory());
    let runtime = shared_core.runtime();
    let token_indices = runtime
        .array_from_i32(&prompt_token_ids, &[1, prompt_token_ids.len() as i32])
        .expect("the token indices upload");
    let embedding_rows = shared_core
        .embedding_lookup(&token_indices)
        .expect("the shared core embeds the prompt");
    let first_layer = &shared_core.weights.decoder_layer_weights[0];
    let core_normalized_rows = runtime
        .rms_norm(
            &embedding_rows,
            &first_layer.input_normalization_weight,
            RMS_NORM_EPSILON,
        )
        .expect("the shared core normalizes");
    let model_serving::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AttentionWeights::Linear(linear_weights) =
        &first_layer.attention_weights
    else {
        panic!("layer 0 of the folded pack is a DeltaNet layer");
    };
    let core_projected_rows = shared_core
        .quantized_linear(
            &core_normalized_rows,
            &linear_weights.input_queries_keys_values_projection,
        )
        .expect("the shared core projects");
    let core_projected_host_rows = host_rows(runtime, &core_projected_rows);
    let folded_query_key_value_rows: Vec<f32> = folded_projected_rows
        .chunks(folded_projected_rows.len() / TRACED_TOKEN_COUNT)
        .flat_map(|row| row[..QUERY_KEY_VALUE_WIDTH].to_vec())
        .collect();
    let model_serving::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
        &first_layer.mlp_weights
    else {
        panic!("the folded pack is a dense model");
    };
    let core_gate_rows = shared_core
        .quantized_linear(&core_normalized_rows, &dense_mlp_weights.gate_projection)
        .expect("the shared core gates");
    let core_up_rows = shared_core
        .quantized_linear(&core_normalized_rows, &dense_mlp_weights.up_projection)
        .expect("the shared core projects up");
    let core_mlp_intermediate_rows = runtime
        .apply_compiled_swiglu(&shared_core.compiled_swiglu, &core_gate_rows, &core_up_rows)
        .expect("the shared core activates");
    let mut core_state = super::shared_core_parity::empty_shared_core_state(&shared_core);
    let Some(model_serving::AstroDecoderStateCacheState::Composite {
        convolution,
        recurrent,
    }) = core_state.layer_mut(0)
    else {
        panic!("layer 0 of the folded pack holds DeltaNet state");
    };
    let core_attention_rows = shared_core
        .forward_linear_attention(
            &core_normalized_rows,
            TRACED_TOKEN_COUNT as i32,
            0,
            linear_weights,
            convolution,
            recurrent,
            None,
            &mut model_serving::PerformanceAttribution::disabled(),
        )
        .expect("the shared core runs layer 0's attention");
    let folded_attention_rows =
        FOLDED_ATTENTION_ROWS.with(|stored_rows| stored_rows.borrow().clone());
    let core_attention_host_rows = host_rows(runtime, &core_attention_rows);
    eprintln!(
        "[prism-shared-core-by-stage] attention largest_difference={:.6} folded_scale={:.4} core_scale={:.4} core_dtype={:?}",
        largest_difference(&folded_attention_rows, &core_attention_host_rows),
        largest_magnitude(&folded_attention_rows),
        largest_magnitude(&core_attention_host_rows),
        core_attention_rows.dtype(),
    );
    let folded_mlp_intermediate_rows =
        FOLDED_MLP_INTERMEDIATE_ROWS.with(|stored_rows| stored_rows.borrow().clone());
    eprintln!(
        "[prism-shared-core-by-stage] mlp_intermediate largest_difference={:.6} folded_scale={:.4}",
        largest_difference(
            &folded_mlp_intermediate_rows,
            &host_rows(runtime, &core_mlp_intermediate_rows)
        ),
        largest_magnitude(&folded_mlp_intermediate_rows),
    );
    eprintln!(
        "[prism-shared-core-by-stage] normalized largest_difference={:.6} core_dtype={:?} core_weight_dtype={:?}",
        largest_difference(
            &folded_normalized_rows,
            &host_rows(runtime, &core_normalized_rows)
        ),
        core_normalized_rows.dtype(),
        first_layer.input_normalization_weight.dtype(),
    );
    eprintln!(
        "[prism-shared-core-by-stage] query_key_value largest_difference={:.6} folded_scale={:.4} core_scale={:.4} core_dtype={:?} storage_rotated={}",
        largest_difference(&folded_query_key_value_rows, &core_projected_host_rows),
        largest_magnitude(&folded_query_key_value_rows),
        largest_magnitude(&core_projected_host_rows),
        core_projected_rows.dtype(),
        matches!(
            linear_weights.input_queries_keys_values_projection.storage,
            model_serving::qwen3_5_core::model_math::dense_projection::Qwen3_5ProjectionStorage::HadamardRotated(_)
        ),
    );
}
