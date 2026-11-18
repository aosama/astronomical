//! How many bytes a reply token would read from storage under different
//! ways of keeping experts in memory, on a real reply's routes.
//!
//! The probe serves the repository's long Shakespeare request through the
//! attributed in-process engine under the configured MLX limit, records every
//! reply token's experts in every layer, and replays that record against
//! several ways of keeping experts (`expert_keeping_simulation`), each given
//! the bytes of experts the engine itself held. The first half of the reply
//! fills each way's memory; the second half is scored. It also prints how
//! much each layer repeats its recent routes, which is what decides whether
//! a small table can serve it.
//!
//! Routes do not depend on the limit, so the record is the same reply the
//! engine would write with all of its experts in memory; only the budget
//! comes from the limit. Run through `cargo measure-expert-keeping` with a
//! configuration whose `model_directory` names the sparse model and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit.

use std::time::Duration;

use astronomical_runtime_integration::MlxMemoryLimits;
use astronomical_test_harness::test_run::TestRunContext;
use serial_test::serial;
use tokio::time::timeout;

use super::decode_residency_attribution::{
    FOLLOW_UP_REQUEST_ID, MEASURED_REQUEST_ID, ProbeRequest, probe_requests,
};
use super::expert_keeping_simulation::{
    KeepingOutcome, ReplyTrace, distinct_expert_count_by_layer, keep_a_fixed_set_of_the_most_used,
    keep_most_recently_used, keep_most_used_so_far, keep_whole_layers_and_recent_tables,
    layers_by_least_repeated_routes, repeated_route_percent_by_layer,
    table_slots_beside_whole_layers,
};
use crate::serving_acceptance::support::performance_attribution::{
    create_attributed_engine, load_engine_with_progress, run_attributed_generation,
};

const EXPERT_KEEPING_TIMEOUT: Duration = Duration::from_secs(120);
const BYTES_PER_MB: f64 = 1_048_576.0;
const BYTES_PER_GB: f64 = 1_073_741_824.0;
/// Whole-layer counts tried beside tables of recent experts.
const WHOLE_LAYER_COUNTS: [usize; 6] = [0, 4, 8, 12, 16, 20];
/// Tokens looked back over when ranking layers by how little they repeat.
const REPEAT_LOOKBACK_TOKEN_COUNT: usize = 16;

#[tokio::test]
#[ignore = "serves a 10,000-token Shakespeare request on an installed sparse model and replays its routes against ways of keeping experts"]
#[serial]
async fn should_measure_bytes_read_a_token_under_each_way_of_keeping_experts() {
    timeout(EXPERT_KEEPING_TIMEOUT, measure_expert_keeping())
        .await
        .expect("the expert keeping probe must finish within 120 seconds");
}

async fn measure_expert_keeping() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let test_run_configuration = TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model to measure");
    let mlx_limit_bytes = usize::try_from(
        test_run_configuration
            .paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    let attribution_directory =
        tempfile::tempdir().expect("the probe should create an attribution directory");
    let mlx_memory_limits = MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
        .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine(
        &model_directory,
        &attribution_directory.path().join("decode.jsonl"),
        &mlx_memory_limits,
        test_run_configuration
            .paging_experiment
            .streaming_prompt_processing_chunk_tokens,
    );
    load_engine_with_progress(&mut engine, "expert_keeping_model_load").await;
    let mut observed_token_count_before_measured_reply = 0;
    for probe_request in probe_requests(&model_directory) {
        if probe_request.request_id == FOLLOW_UP_REQUEST_ID {
            continue;
        }
        if probe_request.request_id == MEASURED_REQUEST_ID {
            observed_token_count_before_measured_reply = observed_token_count(&engine).await;
        }
        serve(&mut engine, &probe_request, &end_of_sequence_token_ids).await;
    }
    let route_trace = engine
        .observed_expert_route_trace_for_tests()
        .await
        .expect("the engine reports its observed routes");
    let kept_expert_bytes = engine
        .astro_expert_cache_statistics_for_tests()
        .await
        .expect("the engine reports its expert residency")
        .resident_payload_byte_count;
    let mut token_routes = Vec::new();
    for observed_route in &route_trace.token_routes[observed_token_count_before_measured_reply..] {
        let mut layer_routes = Vec::with_capacity(observed_route.len());
        for layer_route in observed_route {
            layer_routes.push(layer_route.clone().unwrap_or_default());
        }
        token_routes.push(layer_routes);
    }
    let reply_trace = ReplyTrace {
        token_routes,
        expert_bytes_by_layer: route_trace.expert_bytes_by_layer,
        experts_per_layer: 256,
    };
    report(&reply_trace, kept_expert_bytes);
}

async fn observed_token_count(
    engine: &astronomical_model_serving::Qwen3_5StreamingEngine,
) -> usize {
    engine
        .observed_expert_route_trace_for_tests()
        .await
        .expect("the engine reports its observed routes")
        .token_routes
        .len()
}

async fn serve(
    engine: &mut astronomical_model_serving::Qwen3_5StreamingEngine,
    probe_request: &ProbeRequest,
    end_of_sequence_token_ids: &[u32],
) {
    let generated_token_ids = run_attributed_generation(
        engine,
        probe_request.request_id,
        &probe_request.prompt_token_ids,
        probe_request.phase_name,
        probe_request.output_token_count,
        end_of_sequence_token_ids,
    )
    .await;
    assert!(
        !generated_token_ids.is_empty(),
        "{} must generate tokens",
        probe_request.phase_name
    );
}

fn report(reply_trace: &ReplyTrace, kept_expert_bytes: u64) {
    let reply_token_count = reply_trace.token_routes.len();
    assert!(
        reply_token_count >= 64,
        "the reply recorded only {reply_token_count} tokens; routes are captured while attribution is on"
    );
    let first_scored_token_index = reply_token_count / 2;
    eprintln!(
        "[expert-keeping] reply tokens {reply_token_count} (first {first_scored_token_index} fill memory, the rest are scored), budget {:.2} GB of experts",
        kept_expert_bytes as f64 / BYTES_PER_GB
    );
    report_layer_repetition(reply_trace);
    let least_repeating_layers =
        layers_by_least_repeated_routes(reply_trace, REPEAT_LOOKBACK_TOKEN_COUNT);
    for whole_layer_count in WHOLE_LAYER_COUNTS {
        let whole_layer_indexes = &least_repeating_layers[..whole_layer_count];
        let table_slot_count =
            table_slots_beside_whole_layers(reply_trace, kept_expert_bytes, whole_layer_indexes);
        report_outcome(
            &format!(
                "{whole_layer_count} layers whole, {table_slot_count} recent experts on each other layer"
            ),
            keep_whole_layers_and_recent_tables(
                reply_trace,
                kept_expert_bytes,
                whole_layer_indexes,
                first_scored_token_index,
            ),
        );
    }
    report_outcome(
        "most recently used, one budget across layers",
        keep_most_recently_used(reply_trace, kept_expert_bytes, first_scored_token_index),
    );
    report_outcome(
        "most used so far, one budget across layers",
        keep_most_used_so_far(reply_trace, kept_expert_bytes, first_scored_token_index),
    );
    report_outcome(
        "fixed set of the most used, learned on the first half",
        keep_a_fixed_set_of_the_most_used(
            reply_trace,
            kept_expert_bytes,
            0..first_scored_token_index,
            first_scored_token_index,
        ),
    );
    report_outcome(
        "fixed set of the most used, known in hindsight",
        keep_a_fixed_set_of_the_most_used(
            reply_trace,
            kept_expert_bytes,
            first_scored_token_index..reply_token_count,
            first_scored_token_index,
        ),
    );
}

fn report_outcome(way_of_keeping: &str, outcome: KeepingOutcome) {
    eprintln!(
        "[expert-keeping] {:5.1}% found in memory, {:6.1} MB read a token: {way_of_keeping}",
        outcome.found_in_memory_percent(),
        outcome.missed_bytes_per_token() / BYTES_PER_MB,
    );
}

fn report_layer_repetition(reply_trace: &ReplyTrace) {
    let repeats_previous_token = repeated_route_percent_by_layer(reply_trace, 1);
    let repeats_recent_tokens =
        repeated_route_percent_by_layer(reply_trace, REPEAT_LOOKBACK_TOKEN_COUNT);
    let distinct_expert_counts = distinct_expert_count_by_layer(reply_trace);
    for layer_index in 0..reply_trace.expert_bytes_by_layer.len() {
        eprintln!(
            "[expert-keeping] layer {layer_index:>2}: {:.2} MB an expert, {:>3} different experts used, {:4.1}% of routes repeat the token before, {:4.1}% one of the {REPEAT_LOOKBACK_TOKEN_COUNT} before",
            reply_trace.expert_bytes_by_layer[layer_index] as f64 / BYTES_PER_MB,
            distinct_expert_counts[layer_index],
            repeats_previous_token[layer_index],
            repeats_recent_tokens[layer_index],
        );
    }
}
