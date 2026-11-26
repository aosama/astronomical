//! Decision journeys for the bounded re-promotion of a request-demoted
//! streaming engine back to the complete-residency resident engine.
//! Every journey asserts the sequence of engine roles that actually
//! served, so the hysteresis and the bounded swap counts are proven end
//! to end.

use super::resident_promotion_support::{
    DEMOTING_PROMPT_TOKEN_COUNT, GIBIBYTE_BYTES, MEDIUM_PROMPT_TOKEN_COUNT, PromotionEngineRole,
    PromotionJourney, PromotionScenario, SMALL_PROMPT_TOKEN_COUNT,
};

#[tokio::test]
async fn should_promote_a_demoted_worker_to_resident_for_a_smaller_request() {
    let mut promotion_journey =
        PromotionJourney::launch(PromotionScenario::resident_startup_with_room_to_promote()).await;

    promotion_journey
        .require_completed_generation(1_001, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_002, SMALL_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Resident
        ],
        "the demoting request must stream and the smaller follow-up must run resident"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 1);
    assert_eq!(promotion_journey.resident_promotion_count(), 1);
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_demote_again_after_a_promotion_when_a_large_request_returns() {
    let mut promotion_journey =
        PromotionJourney::launch(PromotionScenario::resident_startup_with_room_to_promote()).await;

    promotion_journey
        .require_completed_generation(1_011, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_012, SMALL_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_013, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Resident,
            PromotionEngineRole::Streaming,
        ],
        "a promoted worker must still demote when pressure returns"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 2);
    assert_eq!(promotion_journey.resident_promotion_count(), 1);
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_stay_streaming_when_the_incoming_context_equals_the_demotion_context() {
    let mut promotion_journey =
        PromotionJourney::launch(PromotionScenario::resident_startup_with_room_to_promote()).await;

    promotion_journey
        .require_completed_generation(1_021, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_022, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Streaming
        ],
        "an equal total context must not swap back, or the same request would thrash"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 1);
    assert_eq!(promotion_journey.resident_promotion_count(), 0);
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_stay_streaming_when_complete_residency_does_not_fit_the_effective_ceiling() {
    let mut promotion_journey = PromotionJourney::launch(PromotionScenario {
        complete_residency_fits_effective_ceiling: false,
        ..PromotionScenario::resident_startup_with_room_to_promote()
    })
    .await;

    promotion_journey
        .require_completed_generation(1_031, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_032, SMALL_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Streaming
        ]
    );
    assert_eq!(promotion_journey.resident_promotion_count(), 0);
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_stay_streaming_when_a_smaller_request_still_does_not_fit_complete_residency() {
    let mut promotion_journey =
        PromotionJourney::launch(PromotionScenario::resident_startup_with_room_to_promote()).await;

    promotion_journey
        .require_completed_generation(1_061, DEMOTING_PROMPT_TOKEN_COUNT)
        .await;
    promotion_journey
        .require_completed_generation(1_062, MEDIUM_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![
            PromotionEngineRole::Streaming,
            PromotionEngineRole::Streaming
        ],
        "a smaller request that still cannot run resident must keep streaming"
    );
    assert_eq!(
        promotion_journey.resident_promotion_count(),
        0,
        "promoting would only demote the request again: two swaps for no resident token"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 1);
    promotion_journey.close().await;
}

#[tokio::test]
async fn should_not_promote_a_streaming_engine_selected_at_startup() {
    let mut promotion_journey = PromotionJourney::launch(PromotionScenario {
        startup_engine_role: PromotionEngineRole::Streaming,
        ..PromotionScenario::resident_startup_with_room_to_promote()
    })
    .await;

    promotion_journey
        .require_completed_generation(1_051, SMALL_PROMPT_TOKEN_COUNT)
        .await;

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![PromotionEngineRole::Streaming],
        "only a request demotion is reversible in place; startup placement is not"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 0);
    assert_eq!(promotion_journey.resident_promotion_count(), 0);
    promotion_journey.close().await;
}

/// A resident engine short for every request by more than any smaller
/// request frees, as a hosted drafter made it at an 18 GB limit (#2578): the
/// worker stays streaming, with one demotion and no promotion, instead of a
/// promotion and a demotion for every request a few tokens smaller.
#[tokio::test]
async fn should_stay_streaming_when_smaller_requests_cannot_free_the_shortfall() {
    let mut promotion_journey = PromotionJourney::launch(PromotionScenario {
        resident_short_for_every_request_bytes: Some(
            usize::try_from(200 * GIBIBYTE_BYTES).unwrap_or(usize::MAX),
        ),
        ..PromotionScenario::resident_startup_with_room_to_promote()
    })
    .await;

    for (request_number, prompt_token_count) in [
        (1_041, DEMOTING_PROMPT_TOKEN_COUNT),
        (1_042, DEMOTING_PROMPT_TOKEN_COUNT - 1),
        (1_043, MEDIUM_PROMPT_TOKEN_COUNT),
        (1_044, SMALL_PROMPT_TOKEN_COUNT),
    ] {
        promotion_journey
            .require_completed_generation(request_number, prompt_token_count)
            .await;
    }

    assert_eq!(
        promotion_journey.counters.serving_engine_roles(),
        vec![PromotionEngineRole::Streaming; 4],
        "every request streams once the resident engine has shown it cannot fit one"
    );
    assert_eq!(promotion_journey.streaming_retry_count(), 1);
    assert_eq!(promotion_journey.resident_promotion_count(), 0);
    promotion_journey.close().await;
}
