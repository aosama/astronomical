//! The same replies the streaming reply-speed journeys measure, asked of the
//! resident engine under the same `paging_experiment` limit, so the two
//! engines can be compared reply for reply before every text model moves to
//! one engine (issue #2585).
//!
//! Run beside the streaming cases on a model with no drafter, so neither side
//! drafts: `cargo measure-resident-reply-speed --configuration PATH` and
//! `cargo measure-hosted-drafter-reply --configuration PATH` (whose silent
//! cases are the streaming side). Alternate the two; one run of either is not
//! a result. Each case serves a warm-up of 500 tokens in and 50 out, one
//! unmeasured reply, then the measured 10,000 in and 500 out. The unmeasured
//! reply is kept although nothing drafts here, so the resident case asks
//! exactly what the streaming silent case asks.
//!
//! Compare reply speed only: the resident engine reads prompts in 2,048-token
//! chunks, the production default, while the streaming side reads the
//! configured streaming chunk, so prompt reading is not like for like here.

use serial_test::serial;
use tokio::time::timeout;

use super::reply_speed_cases::{
    ReplyCase, ReplyTask, RouteLookup, serve_warmup_learning_and_measured_replies,
};
use crate::serving_acceptance::support::performance_attribution::{
    ConfiguredResidentEngine, GenerationSampling, RequestAttribution,
    load_configured_resident_engine,
};

/// Long enough for the resident engine to read a 10,000-token prompt twice
/// and write three replies on a model that fits.
const CASE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(480);

#[tokio::test]
#[ignore = "measures essay reply speed on the resident engine"]
#[serial]
async fn should_measure_essay_reply_speed_on_the_resident_engine() {
    measure_on_the_resident_engine(ReplyTask::Essay, GenerationSampling::Seeded).await;
}

#[tokio::test]
#[ignore = "measures code-temperature reply speed on the resident engine"]
#[serial]
async fn should_measure_code_temperature_reply_speed_on_the_resident_engine() {
    measure_on_the_resident_engine(ReplyTask::Essay, GenerationSampling::SeededForCode).await;
}

#[tokio::test]
#[ignore = "measures code-writing reply speed on the resident engine"]
#[serial]
async fn should_measure_code_writing_reply_speed_on_the_resident_engine() {
    measure_on_the_resident_engine(ReplyTask::Code, GenerationSampling::SeededForCode).await;
}

/// The streaming silent cases' reply, with no drafting and no lookups to
/// choose: the resident engine serves every expert from memory.
async fn measure_on_the_resident_engine(
    reply_task: ReplyTask,
    generation_sampling: GenerationSampling,
) {
    let reply_case = ReplyCase {
        reply_task,
        serves_the_prompt_once_before_measuring: true,
        generation_sampling,
        draft_token_count: 0,
        route_lookup: RouteLookup::InsideTheGraph,
        hosted_drafter_writes_replies: false,
        request_attribution: RequestAttribution::Disabled,
        phase_name: "resident_engine".to_owned(),
    };
    timeout(CASE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let ConfiguredResidentEngine {
            mut engine,
            end_of_sequence_token_ids,
            model_directory,
        } = load_configured_resident_engine("resident_engine_model_load").await;
        let reply_token_ids = serve_warmup_learning_and_measured_replies(
            &mut engine,
            &reply_case,
            &model_directory,
            &end_of_sequence_token_ids,
        )
        .await;
        assert!(
            !reply_token_ids.is_empty(),
            "the resident reply must produce tokens"
        );
    })
    .await
    .expect("the resident reply speed case must finish within its budget");
}
