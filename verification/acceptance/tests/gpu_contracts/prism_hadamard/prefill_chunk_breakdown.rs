//! Where one prompt chunk's GPU time goes on a Qwen 3.5 core pack: a
//! folded ternary pack such as Bonsai, or any other pack the shared core
//! loads, such as an Ornith. It lives beside the folded ternary journeys to
//! share their core loader and Romeo and Juliet prompts.
//!
//! Prompt reading is one forward a 2,048-token chunk, and a prompt forward
//! finishes each layer before the next (`LayerSubmission::PromptLayer`), so
//! a chunk is one evaluation a layer. The profiler therefore answers two
//! questions here:
//!
//! - Per kernel: one linear-attention layer and one full-attention layer,
//!   each a fresh graph over the chunk's embedded rows from an empty decoder
//!   state, timed node by node (`measure_job`).
//! - Between kernels: the whole chunk as serving runs it, timed whole with
//!   the GPU's busy time and its waits on the host (`compare_procedures`
//!   against itself, so the two halves of the report bound the spread).
//!
//! Writes the profiler's pages and JSON under
//! `target/tmp/prefill-chunk-breakdown/` and prints a summary a part.
//! Asserts nothing.

use std::path::PathBuf;

use mlx_bridge::MlxArray;
use mlx_performance_profiler::{
    MeasuredJob, MeasuredProcedure, ReportLocation, compare_procedures, measure_job,
};
use model_serving::{
    AstroDecoderStateCacheState, PerformanceAttribution, Qwen3_5MoEPagedExecutionMode,
    Qwen3_5TextModel, RequestDecoderStateStack,
};
use serial_test::serial;

use super::shared_core::{empty_shared_core_state, loaded_shared_core};
use super::support::{
    PREFILL_CHUNK_TOKENS, pack_directory, pack_tokenizer, romeo_and_juliet_prompt_token_ids,
};

const REPORT_SUBDIRECTORY: &str = "prefill-chunk-breakdown";
/// The whole-chunk journey reads a short chunk so each of its timed runs
/// takes about a second: the pause between layers is the time to encode the
/// next layer, which does not grow with the chunk, so a short chunk shows
/// it larger, not smaller.
const SHORT_CHUNK_TOKENS: usize = 512;

/// One decoder layer over the chunk's rows, from an empty decoder state.
struct OneLayerForward<'model> {
    name: String,
    shared_core: &'model Qwen3_5TextModel,
    layer_index: usize,
    embedded_rows: &'model MlxArray,
    token_count: usize,
}

impl MeasuredJob for OneLayerForward<'_> {
    fn name(&self) -> &str {
        &self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let mut decoder_state = empty_shared_core_state(self.shared_core);
        let layer_output = self
            .one_layer_output(&mut decoder_state)
            .map_err(|forward_error| forward_error.to_string())?;
        // The output reaches the layer's attention writes (attention reads
        // the updated keys and values) and its recurrent kernel; only the
        // small copy of the convolution input's last rows is left out.
        Ok(vec![layer_output])
    }
}

impl OneLayerForward<'_> {
    fn one_layer_output(
        &self,
        decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, model_serving::TextExecutionError> {
        let forward_layer_weights = self.shared_core.forward_layer_weights(self.layer_index)?;
        let (layer_model_state, qwen4_exp_state) =
            model_serving::qwen4_exp::request_state::layer_and_qwen4_exp_state(
                decoder_state,
                self.layer_index,
            )?;
        self.shared_core.forward_decoder_layer(
            self.embedded_rows,
            self.token_count as i32,
            0,
            self.layer_index,
            forward_layer_weights.decoder_layer_weights(),
            layer_model_state,
            None,
            None,
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            qwen4_exp_state,
            &mut PerformanceAttribution::disabled(),
        )
    }
}

/// The whole chunk as serving reads it: every layer, one evaluation each.
struct WholeChunkForward<'model> {
    name: &'static str,
    shared_core: &'model Qwen3_5TextModel,
    chunk_token_ids: &'model [u32],
    decoder_state: Option<RequestDecoderStateStack>,
}

impl MeasuredProcedure for WholeChunkForward<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.decoder_state = Some(empty_shared_core_state(self.shared_core));
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        forward_prepared_chunk(
            self.shared_core,
            self.chunk_token_ids,
            &mut self.decoder_state,
        )
        .map(|_final_logits| ())
    }
}

/// One chunk through every layer from the decoder state a procedure's
/// `prepare` left, returning the chunk's final logits.
pub(super) fn forward_prepared_chunk(
    shared_core: &Qwen3_5TextModel,
    chunk_token_ids: &[u32],
    decoder_state: &mut Option<RequestDecoderStateStack>,
) -> Result<MlxArray, String> {
    let decoder_state = decoder_state
        .as_mut()
        .ok_or("the decoder state was not prepared")?;
    shared_core
        .forward_chunk(chunk_token_ids, 0, decoder_state)
        .map_err(|forward_error| forward_error.to_string())
}

#[test]
#[serial]
#[ignore = "loads an installed Qwen 3.5 core pack on the GPU; run through cargo measure-prefill-chunk-breakdown with model_directory configured"]
fn should_break_down_one_prompt_chunk_layer_by_kernel() {
    let shared_core = loaded_shared_core(&pack_directory());
    let chunk_token_ids = chunk_token_ids(PREFILL_CHUNK_TOKENS);
    let embedded_rows = embedded_rows(&shared_core, &chunk_token_ids);
    let report_directory = report_directory();
    std::fs::create_dir_all(&report_directory).expect("the report folder is created");
    for (layer_kind, layer_index) in one_layer_of_each_kind(&shared_core) {
        measure_one_layer(
            (&shared_core, &embedded_rows, PREFILL_CHUNK_TOKENS),
            layer_kind,
            layer_index,
            &report_directory,
        );
    }
}

#[test]
#[serial]
#[ignore = "loads an installed Qwen 3.5 core pack on the GPU; run through cargo measure-prefill-between-layers with model_directory configured"]
/// The chunk's time less its layers' times is the time between layers only
/// on a dense pack. A mixture-of-experts layer timed alone pays its expert
/// page load and a route stop that serving hides behind its read-ahead, so on
/// such a pack read the whole chunk's GPU idle time instead.
fn should_measure_the_time_a_chunk_spends_between_layers() {
    let shared_core = loaded_shared_core(&pack_directory());
    let chunk_token_ids = chunk_token_ids(SHORT_CHUNK_TOKENS);
    let embedded_rows = embedded_rows(&shared_core, &chunk_token_ids);
    let report_directory = report_directory();
    std::fs::create_dir_all(&report_directory).expect("the report folder is created");
    let mut layers_summed_milliseconds = 0.0;
    for (layer_kind, layer_index) in one_layer_of_each_kind(&shared_core) {
        let layer_milliseconds = measure_one_layer(
            (&shared_core, &embedded_rows, SHORT_CHUNK_TOKENS),
            layer_kind,
            layer_index,
            &report_directory,
        );
        layers_summed_milliseconds +=
            layer_milliseconds * layer_count_of_kind(&shared_core, layer_kind) as f64;
    }
    let chunk_milliseconds = measure_whole_chunk(&shared_core, &chunk_token_ids, &report_directory);
    println!(
        "[prefill-chunk-breakdown] part=between_layers tokens={SHORT_CHUNK_TOKENS} chunk_ms={chunk_milliseconds:.1} layers_alone_ms={layers_summed_milliseconds:.1} between_layers_ms={:.1} share={:.1}%",
        chunk_milliseconds - layers_summed_milliseconds,
        100.0 * (chunk_milliseconds - layers_summed_milliseconds) / chunk_milliseconds,
    );
}

#[test]
#[serial]
#[ignore = "loads an installed pack on the GPU; run through cargo measure-prefill-served-chunk with model_directory configured"]
fn should_measure_a_served_chunk_whole() {
    let shared_core = loaded_shared_core(&pack_directory());
    let chunk_token_ids = chunk_token_ids(PREFILL_CHUNK_TOKENS);
    let report_directory = report_directory();
    std::fs::create_dir_all(&report_directory).expect("the report folder is created");
    measure_whole_chunk(&shared_core, &chunk_token_ids, &report_directory);
}

/// The layer's median wall time over the profiler's runs, in milliseconds.
fn measure_one_layer(
    chunk: (&Qwen3_5TextModel, &MlxArray, usize),
    layer_kind: &str,
    layer_index: usize,
    report_directory: &std::path::Path,
) -> f64 {
    let (shared_core, embedded_rows, token_count) = chunk;
    let mut one_layer_forward = OneLayerForward {
        name: format!("{layer_kind}-layer-{layer_index}"),
        shared_core,
        layer_index,
        embedded_rows,
        token_count,
    };
    let file_base = format!("prefill-{layer_kind}-layer-{token_count}-tokens");
    let job_summary = measure_job(
        &mut one_layer_forward,
        ReportLocation {
            title: "One decoder layer over a prompt chunk",
            directory: report_directory,
            file_base: &file_base,
        },
    )
    .expect("the profiler measures the layer");
    println!(
        "[prefill-chunk-breakdown] part={layer_kind}_layer layer_index={layer_index} wall_median_ms={:.2} gpu_execution_ms={:.2} nodes={} report={file_base}.json",
        job_summary.job_wall_median_microseconds / 1_000.0,
        job_summary.gpu_execution_microseconds / 1_000.0,
        job_summary.executed_node_count,
    );
    job_summary.job_wall_median_microseconds / 1_000.0
}

/// The chunk's median wall time over the profiler's runs, in milliseconds.
fn measure_whole_chunk(
    shared_core: &Qwen3_5TextModel,
    chunk_token_ids: &[u32],
    report_directory: &std::path::Path,
) -> f64 {
    let mut first_copy = WholeChunkForward {
        name: "whole-chunk-a",
        shared_core,
        chunk_token_ids,
        decoder_state: None,
    };
    let mut second_copy = WholeChunkForward {
        name: "whole-chunk-b",
        shared_core,
        chunk_token_ids,
        decoder_state: None,
    };
    let comparison = compare_procedures(
        &mut first_copy,
        &mut second_copy,
        ReportLocation {
            title: "One short prompt chunk, every layer",
            directory: report_directory,
            file_base: "prefill-whole-chunk",
        },
    )
    .expect("the profiler measures the chunk");
    println!(
        "[prefill-chunk-breakdown] part=whole_chunk duration_ms={:.1} gpu_busy_ms={:.1} gpu_idle_ms={:.1} blocking_waits={:.0} command_buffers={:.0} launches={}",
        comparison.duration.baseline_median_microseconds / 1_000.0,
        comparison.gpu_busy.baseline_median_microseconds / 1_000.0,
        comparison.gpu_idle.baseline_median_microseconds / 1_000.0,
        comparison.baseline.median_blocking_wait_count,
        comparison.baseline.median_command_buffer_count,
        comparison.baseline.kernel_launch_count,
    );
    comparison.duration.baseline_median_microseconds / 1_000.0
}

/// The first linear-attention and the first full-attention layer.
fn one_layer_of_each_kind(shared_core: &Qwen3_5TextModel) -> Vec<(&'static str, usize)> {
    let decoder_state = empty_shared_core_state(shared_core);
    let layer_states = decoder_state.layers();
    let first_of = |is_attention: bool| {
        layer_states.iter().position(|layer_state| {
            matches!(
                layer_state,
                AstroDecoderStateCacheState::AppendOnlyAttention { .. }
            ) == is_attention
        })
    };
    let mut layers = Vec::new();
    layers.extend(first_of(false).map(|layer_index| ("linear_attention", layer_index)));
    layers.extend(first_of(true).map(|layer_index| ("full_attention", layer_index)));
    layers
}

/// How many of the model's layers are of `layer_kind`.
fn layer_count_of_kind(shared_core: &Qwen3_5TextModel, layer_kind: &str) -> usize {
    let decoder_state = empty_shared_core_state(shared_core);
    let attention_layer_count = decoder_state
        .layers()
        .iter()
        .filter(|layer_state| {
            matches!(
                layer_state,
                AstroDecoderStateCacheState::AppendOnlyAttention { .. }
            )
        })
        .count();
    match layer_kind {
        "full_attention" => attention_layer_count,
        _ => decoder_state.layers().len() - attention_layer_count,
    }
}

pub(super) fn chunk_token_ids(token_count: usize) -> Vec<u32> {
    romeo_and_juliet_prompt_token_ids(&pack_tokenizer(), token_count)
        .into_iter()
        .take(token_count)
        .map(|token_id| token_id as u32)
        .collect()
}

/// The chunk's rows as the first layer reads them, evaluated once and shared
/// by every round.
fn embedded_rows(shared_core: &Qwen3_5TextModel, chunk_token_ids: &[u32]) -> MlxArray {
    let signed_token_ids: Vec<i32> = chunk_token_ids
        .iter()
        .map(|token_id| *token_id as i32)
        .collect();
    let token_indices = shared_core
        .runtime()
        .array_from_i32(&signed_token_ids, &[1, signed_token_ids.len() as i32])
        .expect("the token indices upload");
    let embedded_rows = shared_core
        .context_free_token_meanings(&token_indices)
        .expect("the chunk embeds");
    shared_core
        .runtime()
        .evaluate_arrays(&[&embedded_rows])
        .expect("the embedded rows evaluate");
    embedded_rows
}

fn report_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tmp")
        .join(REPORT_SUBDIRECTORY)
}
