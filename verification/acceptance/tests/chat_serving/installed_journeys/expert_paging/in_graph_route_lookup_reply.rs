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

use super::mtp_reply_drafting_speed::{
    ReplyCase, RouteLookup, run_reply_case, write_compared_greedy_reply,
};
use crate::serving_acceptance::support::performance_attribution::{
    GenerationSampling, PagingExperimentEngine, RequestAttribution, load_paging_experiment_engine,
};

const REFUSING_REQUEST_ID: RequestId = RequestId::new(98_601);
const IN_GRAPH_REQUEST_ID: RequestId = RequestId::new(98_602);
const ON_HOST_REQUEST_ID: RequestId = RequestId::new(98_603);
/// Three drafts and the token before them: a block of four rows.
const PROOF_DRAFT_TOKEN_COUNT: u32 = 3;
const PROOF_TIMEOUT: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "writes one greedy reply both ways on an installed streamed model"]
#[serial]
async fn should_write_the_same_greedy_reply_whichever_way_routes_are_looked_up() {
    timeout(PROOF_TIMEOUT, compare_greedy_replies(0))
        .await
        .expect("the route lookup proof must finish within its budget");
}

/// The same proof for drafted blocks (a model with a multi-token-prediction
/// head): a block of four rows is verified with its routes looked up in the
/// graph, refused and forwarded again in the first reply, and the reply must
/// be the plain reply resolved on the host.
#[tokio::test]
#[ignore = "writes one greedy reply drafted and plain on an installed streamed model with a draft head"]
#[serial]
async fn should_write_the_same_greedy_reply_when_drafted_blocks_are_looked_up_in_the_graph() {
    timeout(
        PROOF_TIMEOUT,
        compare_greedy_replies(PROOF_DRAFT_TOKEN_COUNT),
    )
    .await
    .expect("the drafted block proof must finish within its budget");
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
        hosted_drafter_writes_replies: true,
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
    /// Tokens drafted a round; zero writes the reply one token a forward.
    draft_token_count: u32,
}

/// Writes the reply three ways. The two in-graph replies draft
/// `in_graph_draft_token_count` tokens a round; the reply on the host never
/// drafts, so it is the plain reply both are held to.
async fn compare_greedy_replies(in_graph_draft_token_count: u32) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut paging_engine = load_paging_experiment_engine("route_lookup_model_load").await;
    let refusing_reply = ComparedReply {
        request_id: REFUSING_REQUEST_ID,
        route_lookup: RouteLookup::InsideTheGraph,
        grows_tables_ahead_of_lookups: false,
        draft_token_count: in_graph_draft_token_count,
    };
    let reply_as_served = ComparedReply {
        request_id: IN_GRAPH_REQUEST_ID,
        route_lookup: RouteLookup::InsideTheGraph,
        grows_tables_ahead_of_lookups: true,
        draft_token_count: in_graph_draft_token_count,
    };
    let reply_on_host = ComparedReply {
        request_id: ON_HOST_REQUEST_ID,
        route_lookup: RouteLookup::OnTheHost,
        grows_tables_ahead_of_lookups: true,
        draft_token_count: 0,
    };
    // Without drafting the lookups go first: they then meet tables no exact
    // reply has filled, so they have experts to read and tables to grow.
    // With drafting the plain reply goes first, because the draft head joins
    // the memory ledger when drafting is turned on and cannot leave it under
    // tables a drafted reply has already filled to the limit; a block's rows
    // still need more room than a plain reply grew.
    let (reference_reply, refusing_reply_tokens, reply_as_served_tokens, refused_token_counts) =
        if in_graph_draft_token_count == 0 {
            let refusing_tokens = write_greedy_reply(&mut paging_engine, &refusing_reply).await;
            let refused_after_refusing = refused_token_count(&paging_engine).await;
            let as_served_tokens = write_greedy_reply(&mut paging_engine, &reply_as_served).await;
            let refused_after_as_served = refused_token_count(&paging_engine).await;
            let reference = write_greedy_reply(&mut paging_engine, &reply_on_host).await;
            (
                reference,
                refusing_tokens,
                as_served_tokens,
                [0, refused_after_refusing, refused_after_as_served],
            )
        } else {
            let reference = write_greedy_reply(&mut paging_engine, &reply_on_host).await;
            let refused_before = refused_token_count(&paging_engine).await;
            let refusing_tokens = write_greedy_reply(&mut paging_engine, &refusing_reply).await;
            let refused_after_refusing = refused_token_count(&paging_engine).await;
            let as_served_tokens = write_greedy_reply(&mut paging_engine, &reply_as_served).await;
            let refused_after_as_served = refused_token_count(&paging_engine).await;
            (
                reference,
                refusing_tokens,
                as_served_tokens,
                [
                    refused_before,
                    refused_after_refusing,
                    refused_after_as_served,
                ],
            )
        };
    let in_graph_counts = paging_engine
        .engine
        .in_graph_serving_counts_for_tests()
        .await
        .expect("the engine counts its lookups");
    eprintln!(
        "[route-lookup-proof] draft_tokens={in_graph_draft_token_count} in_graph_counts={in_graph_counts:?} refused_before_refusing_and_as_served={refused_token_counts:?} reply_tokens={}",
        reference_reply.len()
    );
    assert!(
        in_graph_counts.served_layer_count > 0 && in_graph_counts.fetching_layer_count > 0,
        "lookups must have served layers and read experts for the comparison to prove anything: {in_graph_counts:?}"
    );
    assert!(
        refused_token_counts[1] > refused_token_counts[0],
        "the refusing reply's lookups must have refused for it to prove the re-run"
    );
    assert_eq!(
        refused_token_counts[2], refused_token_counts[1],
        "with tables grown ahead no lookup refuses"
    );
    assert!(
        reference_reply.len() > 1,
        "the compared reply must be more than one token"
    );
    assert_eq!(
        refusing_reply_tokens, reference_reply,
        "a reply with refused forwards served again must be the reply resolved on the host"
    );
    assert_eq!(
        reply_as_served_tokens, reference_reply,
        "the reply looked up inside the graph must be the reply resolved on the host"
    );
}

async fn refused_token_count(paging_engine: &PagingExperimentEngine) -> u64 {
    paging_engine
        .engine
        .in_graph_serving_counts_for_tests()
        .await
        .expect("the engine counts its lookups")
        .refused_token_count
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
        .set_multi_token_prediction_draft_tokens_for_tests(compared_reply.draft_token_count)
        .await
        .expect("the engine takes the draft count");
    paging_engine
        .engine
        .set_tables_grown_ahead_of_lookups_for_tests(compared_reply.grows_tables_ahead_of_lookups)
        .await
        .expect("the engine takes the table growth");
    write_compared_greedy_reply(
        paging_engine,
        compared_reply.request_id,
        &phase_name(compared_reply.route_lookup),
    )
    .await
}
