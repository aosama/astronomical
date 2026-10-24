//! Failure journeys for resident re-promotion: a promoted engine that
//! fails to load, a resident creation failure that must leave the
//! streaming engine serving, and a swap-only runtime without a startup
//! engine placement.

use super::resident_promotion_support::{
    DEMOTING_PROMPT_TOKEN_COUNT, PromotionEngineRole, PromotionJourney, PromotionScenario,
    SMALL_PROMPT_TOKEN_COUNT,
};
use astronomical_ipc_protocol::{ChatGenerationFailureReason, WorkerEvent};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn should_report_a_fatal_failure_when_the_promoted_engine_fails_to_load() {
    let mut promotion_journey = PromotionJourney::launch(PromotionScenario {
        resident_promotion_load_succeeds: false,
        ..PromotionScenario::resident_startup_with_room_to_promote()
    })
    .await;

    promotion_journey
        .require_completed_generation(1_071, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    let failure_reason = promotion_journey
        .require_failed_generation(1_072, SMALL_PROMPT_TOKEN_COUNT)
        .await;

    assert!(
        matches!(
            failure_reason,
            ChatGenerationFailureReason::FatalExecution { .. }
        ),
        "a load failure after the streaming engine was released leaves no runtime: {failure_reason:?}"
    );
    assert_eq!(promotion_journey.resident_promotion_count(), 1);
    let worker_outcome = timeout(Duration::from_secs(5), promotion_journey.worker_task)
        .await
        .expect("the worker should exit after a fatal promotion failure")
        .expect("the worker task should not panic");
    assert!(
        worker_outcome.is_err(),
        "the worker must exit with an error once it has no runtime"
    );
}

#[tokio::test]
async fn should_keep_serving_on_streaming_when_resident_promotion_creation_fails() {
    let mut promotion_journey = PromotionJourney::launch(PromotionScenario {
        resident_promotion_creation_succeeds: false,
        ..PromotionScenario::resident_startup_with_room_to_promote()
    })
    .await;

    promotion_journey
        .require_completed_generation(1_041, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_042, SMALL_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Streaming
        ],
        "a failed resident creation must leave the streaming engine serving"
    );
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_reject_a_model_swap_whose_runtime_carries_no_startup_placement() {
    let mut promotion_journey = PromotionJourney::launch_idle(PromotionScenario {
        startup_runtime_carries_engine_placement: false,
        ..PromotionScenario::resident_startup_with_room_to_promote()
    });

    let model_swap_event = promotion_journey.request_model_swap().await;

    assert_eq!(
        model_swap_event,
        WorkerEvent::ModelSwapFailed {
            loaded_model_remains_ready: false,
            model_load_failure_reason:
                "model factory returned a swap-only runtime without a startup engine placement"
                    .to_owned(),
        },
        "only a worker-directed swap may load a runtime without a startup placement"
    );
    promotion_journey.close().await;
}
