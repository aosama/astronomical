//! A streamed reply whose layers look their routes up inside each token's
//! graph, against the same reply with every route copied to the host
//! (`in_graph_table_serving` in the qwen35 family, issue #2533).
//!
//! The first journey is the proof: one loaded model writes one greedy reply
//! three times and the tokens must be the same each time. First with
//! lookups and no table grown between tokens, so some lookups refuse and
//! their tokens are served again the exact way; then with lookups as the
//! engine serves, when none may refuse; then with every route on the host.
//! The lookups are shown to have served layers and read experts from
//! storage. The others are the
//! measurement, one process a case so neither way starts from tables the
//! other warmed: a 500/50 warmup, then a 10,000-token Romeo and Juliet
//! summary and 500 reply tokens through the engine's own reply loop.
//! Compare the cases' `[performance-attribution] status=success` lines.
//!
//! Run with `cargo test-in-graph-route-lookup --configuration PATH` and
//! `cargo measure-in-graph-route-lookup --configuration PATH`, whose
//! `model_directory` names the model and whose
//! `paging_experiment.maximum_mlx_memory_gigabytes` sets the limit. About
//! fifty seconds for the proof and seventy a measured case.

use std::time::Duration;

use ipc_protocol::RequestId;
use model_serving::Qwen3_5StreamingEngineTestControls;
use serial_test::serial;
use tokio::time::timeout;

use super::mtp_reply_drafting_speed::{ReplyCase, RouteLookup, run_reply_case};
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, PagingExperimentEngine, RequestAttribution, load_paging_experiment_engine,
    run_generation_with_request_attribution,
};

const REFUSING_REQUEST_ID: RequestId = RequestId::new(98_601);
const IN_GRAPH_REQUEST_ID: RequestId = RequestId::new(98_602);
const ON_HOST_REQUEST_ID: RequestId = RequestId::new(98_603);
/// Past the sparse-attention budget of a lane-mixing model, and short
/// enough to read twice in the proof's budget.
const COMPARED_PROMPT_TOKEN_COUNT: usize = 2_600;
const COMPARED_OUTPUT_TOKEN_COUNT: u16 = 96;
const PROOF_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "writes one greedy reply both ways on an installed streamed model"]
#[serial]
async fn should_write_the_same_greedy_reply_whichever_way_routes_are_looked_up() {
    timeout(PROOF_TIMEOUT, compare_greedy_replies())
        .await
        .expect("the route lookup proof must finish within its budget");
}

#[tokio::test]
#[ignore = "measures essay reply speed with every route copied to the host"]
#[serial]
async fn should_measure_essay_reply_speed_with_routes_resolved_on_the_host() {
    run_reply_case(speed_case(
        GenerationSampling::Seeded,
        RouteLookup::OnTheHost,
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures essay reply speed with routes looked up inside the graph"]
#[serial]
async fn should_measure_essay_reply_speed_with_routes_looked_up_in_the_graph() {
    run_reply_case(speed_case(
        GenerationSampling::Seeded,
        RouteLookup::InsideTheGraph,
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed with every route copied to the host"]
#[serial]
async fn should_measure_code_temperature_reply_speed_with_routes_resolved_on_the_host() {
    run_reply_case(speed_case(
        GenerationSampling::SeededForCode,
        RouteLookup::OnTheHost,
    ))
    .await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed with routes looked up inside the graph"]
#[serial]
async fn should_measure_code_temperature_reply_speed_with_routes_looked_up_in_the_graph() {
    run_reply_case(speed_case(
        GenerationSampling::SeededForCode,
        RouteLookup::InsideTheGraph,
    ))
    .await;
}

#[tokio::test]
#[ignore = "attributes an essay reply's time with every route copied to the host"]
#[serial]
async fn should_attribute_essay_reply_time_with_routes_resolved_on_the_host() {
    run_reply_case(attributed_case(RouteLookup::OnTheHost)).await;
}

#[tokio::test]
#[ignore = "attributes an essay reply's time with routes looked up inside the graph"]
#[serial]
async fn should_attribute_essay_reply_time_with_routes_looked_up_in_the_graph() {
    run_reply_case(attributed_case(RouteLookup::InsideTheGraph)).await;
}

fn phase_name(route_lookup: RouteLookup) -> String {
    match route_lookup {
        RouteLookup::InsideTheGraph => "routes_in_graph".to_owned(),
        RouteLookup::OnTheHost => "routes_on_host".to_owned(),
    }
}

/// A reply written one token a forward, so the route lookup is the only
/// thing the two cases differ in.
fn speed_case(generation_sampling: GenerationSampling, route_lookup: RouteLookup) -> ReplyCase {
    ReplyCase {
        generation_sampling,
        draft_token_count: 0,
        route_lookup,
        request_attribution: RequestAttribution::Disabled,
        phase_name: phase_name(route_lookup),
    }
}

/// Attribution slows the reply, so only the shares of its time are to be
/// read from this case, not its speed.
fn attributed_case(route_lookup: RouteLookup) -> ReplyCase {
    ReplyCase {
        request_attribution: RequestAttribution::Enabled,
        ..speed_case(GenerationSampling::Seeded, route_lookup)
    }
}

/// One greedy reply of the proof: which request, and how its routes are
/// looked up.
struct ComparedReply {
    request_id: RequestId,
    route_lookup: RouteLookup,
    /// Off for the reply whose lookups are to refuse: with no table grown
    /// between tokens, a route that needs a larger one is refused and its
    /// token is served again the exact way.
    grows_tables_ahead_of_lookups: bool,
}

async fn compare_greedy_replies() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut paging_engine = load_paging_experiment_engine("route_lookup_model_load").await;
    let mut greedy_replies = Vec::with_capacity(3);
    let mut refused_token_counts = Vec::with_capacity(3);
    // The lookups go first: they then meet tables no exact reply has filled,
    // so they have experts to read and, in the first reply, tables to grow.
    for compared_reply in [
        ComparedReply {
            request_id: REFUSING_REQUEST_ID,
            route_lookup: RouteLookup::InsideTheGraph,
            grows_tables_ahead_of_lookups: false,
        },
        ComparedReply {
            request_id: IN_GRAPH_REQUEST_ID,
            route_lookup: RouteLookup::InsideTheGraph,
            grows_tables_ahead_of_lookups: true,
        },
        ComparedReply {
            request_id: ON_HOST_REQUEST_ID,
            route_lookup: RouteLookup::OnTheHost,
            grows_tables_ahead_of_lookups: true,
        },
    ] {
        greedy_replies.push(write_greedy_reply(&mut paging_engine, &compared_reply).await);
        refused_token_counts.push(
            paging_engine
                .engine
                .in_graph_serving_counts_for_tests()
                .await
                .expect("the engine counts its lookups")
                .refused_token_count,
        );
    }
    let in_graph_counts = paging_engine
        .engine
        .in_graph_serving_counts_for_tests()
        .await
        .expect("the engine counts its lookups");
    eprintln!(
        "[route-lookup-proof] in_graph_counts={in_graph_counts:?} refused_tokens_after_each_reply={refused_token_counts:?} reply_tokens={}",
        greedy_replies[0].len()
    );
    assert!(
        in_graph_counts.served_layer_count > 0 && in_graph_counts.fetching_layer_count > 0,
        "lookups must have served layers and read experts for the comparison to prove anything: {in_graph_counts:?}"
    );
    assert!(
        refused_token_counts[0] > 0,
        "the first reply's lookups must have refused tokens for it to prove the re-run"
    );
    assert_eq!(
        refused_token_counts[1], refused_token_counts[0],
        "with tables grown between tokens no lookup refuses"
    );
    assert!(
        greedy_replies[0].len() > 1,
        "the compared reply must be more than one token"
    );
    assert_eq!(
        greedy_replies[0], greedy_replies[2],
        "a reply with refused tokens served again must be the reply resolved on the host"
    );
    assert_eq!(
        greedy_replies[1], greedy_replies[2],
        "the reply looked up inside the graph must be the reply resolved on the host"
    );
}

/// Sets the engine up as `compared_reply` says and writes its greedy reply.
async fn write_greedy_reply(
    paging_engine: &mut PagingExperimentEngine,
    compared_reply: &ComparedReply,
) -> Vec<u32> {
    paging_engine
        .engine
        .set_routes_looked_up_in_the_graph_for_tests(
            compared_reply.route_lookup == RouteLookup::InsideTheGraph,
        )
        .await
        .expect("the engine takes the route lookup");
    paging_engine
        .engine
        .set_tables_grown_ahead_of_lookups_for_tests(compared_reply.grows_tables_ahead_of_lookups)
        .await
        .expect("the engine takes the table growth");
    let model_id = paging_engine
        .model_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the model directory has a UTF-8 name")
        .to_owned();
    let prompt_token_ids =
        crate::serving_acceptance::support::romeo_and_juliet::prepare_romeo_and_juliet_three_paragraph_summary_prompt(
            &paging_engine.model_directory,
            &model_id,
            compared_reply.request_id,
            COMPARED_PROMPT_TOKEN_COUNT,
            COMPARED_OUTPUT_TOKEN_COUNT,
        );
    run_generation_with_request_attribution(
        &mut paging_engine.engine,
        compared_reply.request_id,
        &prompt_token_ids,
        &phase_name(compared_reply.route_lookup),
        COMPARED_OUTPUT_TOKEN_COUNT,
        &paging_engine.end_of_sequence_token_ids,
        GenerationSampling::Greedy,
        RequestAttribution::Disabled,
    )
    .await
}
