//! Acceptance tests for the stop-first memory-demotion worker restart.
//!
//! The contract these tests pin is that the trusted worker process has exited
//! and been reaped before any replacement process is launched. Asserting that
//! from the supervisor's own control flow would only restate the implementation,
//! so the ordering is observed from the replacement process itself: the worker
//! fixture records the liveness of every sibling worker at the instant it
//! starts. A stop-first restart therefore records its predecessor as gone, and a
//! candidate-first restart would record it as alive.

use std::sync::Arc;
use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, MlxMemoryLimitUpdateOutcome, WorkerHealthStatus,
};
use tokio::time::timeout;

use super::worker_memory_demotion_support::{
    DemotionTestContext, JOURNEY_TIMEOUT, LAUNCH_ORDER_LOG_FILE, WORKER_PROCESS_IDENTIFIER_PREFIX,
    assert_launch_observation, demotion_fixture_path, model_policy_catalog, process_is_running,
    recorded_worker_identifier, startup_configuration,
};

const TRUSTED_CONFIGURATION_GENERATION: &str = "trusted-generation";
const REPLACEMENT_CONFIGURATION_GENERATION: &str = "replacement-generation";
const TRUSTED_MODEL_ID: &str = "astronomical/trusted-model";
const RESTORE_FAILURE_MODEL_ID: &str = "astronomical/restore-failure-model";
const SILENT_CANDIDATE_GENERATION: &str = "silent-acknowledgement-generation";
const MISMATCHED_CANDIDATE_GENERATION: &str = "mismatched-acknowledgement-generation";
const IN_PLACE_TRUSTED_GENERATION: &str = "in-place-trusted-generation";
const IN_PLACE_REDUCED_GENERATION: &str = "in-place-reduced-generation";
const IN_PLACE_RAISED_GENERATION: &str = "in-place-raised-generation";

const DEMOTION_CEILING_BYTES: u64 = 19_000_000_000;

#[tokio::test]
async fn should_reap_the_trusted_worker_before_launching_the_demotion_replacement() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget");
    let restored_configuration = demotion_outcome.unwrap_or_else(|demotion_error| {
        panic!(
            "the demotion restart should restore the candidate configuration: {demotion_error:?}; worker_health={:?}",
            demotion_context.worker_handle.worker_health_snapshot()
        )
    });

    assert_eq!(
        restored_configuration.configuration_generation,
        REPLACEMENT_CONFIGURATION_GENERATION
    );
    assert_launch_observation(
        &demotion_context.temporary_directory,
        REPLACEMENT_CONFIGURATION_GENERATION,
        TRUSTED_CONFIGURATION_GENERATION,
        "gone",
    );
    assert!(
        !process_is_running(&recorded_worker_identifier(
            &demotion_context.temporary_directory,
            TRUSTED_CONFIGURATION_GENERATION,
        )),
        "the trusted worker process must be reaped once the demotion reports success"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_restore_the_selected_model_before_reporting_demotion_success() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget")
    .expect("the demotion restart should succeed for a restorable model");

    let restored_health = demotion_context.worker_handle.worker_health_snapshot();
    assert_eq!(
        restored_health.status,
        WorkerHealthStatus::Ready,
        "success must not be reported while the worker is not serving"
    );
    assert_eq!(
        restored_health.ready_model_id.as_deref(),
        Some(TRUSTED_MODEL_ID),
        "the same selected model must be restored before the endpoint reports success"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_end_the_demotion_in_idle_when_no_model_had_been_selected() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;

    let restored_configuration = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the stop-first restart should complete inside the journey budget")
    .expect("an idle demotion should succeed without an unnecessary model load");

    assert_eq!(
        restored_configuration.configuration_generation,
        REPLACEMENT_CONFIGURATION_GENERATION
    );
    assert_eq!(
        demotion_context
            .worker_handle
            .worker_health_snapshot()
            .ready_model_id,
        None,
        "a demotion with no prior selection must not invent one"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_not_publish_readiness_when_the_selected_model_cannot_be_restored() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, RESTORE_FAILURE_MODEL_ID)
            .await;
    demotion_context
        .load_selected_model(RESTORE_FAILURE_MODEL_ID)
        .await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(RESTORE_FAILURE_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the restart attempt should complete inside the journey budget");

    assert!(
        demotion_outcome.is_err(),
        "a failed model restoration must not report demotion success"
    );
    let restored_health = demotion_context.worker_handle.worker_health_snapshot();
    assert_ne!(
        restored_health.ready_model_id.as_deref(),
        Some(RESTORE_FAILURE_MODEL_ID),
        "a model the replacement never loaded must never be published as ready"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_a_candidate_that_never_acknowledges_its_configuration() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    SILENT_CANDIDATE_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the rejection should complete inside the journey budget");

    // The contract under test is that a silent candidate is never accepted:
    // which boundary reports the failure first (the acknowledgement timeout
    // here, or trusted-worker containment when a loaded runner starves the
    // handshake) is scheduler-dependent, so the post-condition is deliberately
    // variant-independent.
    assert!(
        demotion_outcome.is_err(),
        "a silent candidate must never be accepted, received {demotion_outcome:?}"
    );
    assert_eq!(
        demotion_context
            .worker_handle
            .worker_health_snapshot()
            .status,
        WorkerHealthStatus::Unavailable,
        "a rejected demotion must publish a truthful unavailable state"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_reject_a_candidate_that_acknowledges_an_unrequested_generation() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_fixture_path(),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    MISMATCHED_CANDIDATE_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the rejection should complete inside the journey budget");

    assert!(
        demotion_outcome.is_err(),
        "candidate configuration identity must bind, received {demotion_outcome:?}"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_publish_unavailable_when_the_replacement_executable_cannot_launch() {
    let demotion_context =
        DemotionTestContext::launch(TRUSTED_CONFIGURATION_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;

    let demotion_outcome = timeout(
        JOURNEY_TIMEOUT,
        demotion_context
            .worker_handle
            .restart_worker_for_memory_demotion(
                demotion_context
                    .temporary_directory
                    .join("missing-replacement-executable"),
                model_policy_catalog(TRUSTED_MODEL_ID),
                startup_configuration(
                    REPLACEMENT_CONFIGURATION_GENERATION,
                    &demotion_context.temporary_directory,
                    Some(DEMOTION_CEILING_BYTES),
                ),
            ),
    )
    .await
    .expect("the failed launch should be reported inside the journey budget");

    assert!(
        demotion_outcome.is_err(),
        "a candidate that cannot launch must fail the demotion, received {demotion_outcome:?}"
    );
    assert_eq!(
        demotion_context
            .worker_handle
            .worker_health_snapshot()
            .status,
        WorkerHealthStatus::Unavailable,
        "a demotion whose replacement cannot launch must publish a truthful unavailable state"
    );
    let launch_order_log = std::fs::read_to_string(
        demotion_context
            .temporary_directory
            .join(LAUNCH_ORDER_LOG_FILE),
    )
    .expect("the trusted worker should have recorded its launch observation");
    assert!(
        !launch_order_log
            .lines()
            .any(|line| line
                .starts_with(&format!("{REPLACEMENT_CONFIGURATION_GENERATION} observed"))),
        "no replacement may launch when the candidate executable is missing, recorded {launch_order_log:?}"
    );
    assert!(
        !process_is_running(&recorded_worker_identifier(
            &demotion_context.temporary_directory,
            TRUSTED_CONFIGURATION_GENERATION,
        )),
        "the stop-first ordering must hold even when the replacement cannot launch"
    );
    demotion_context.shutdown().await;
}

#[tokio::test]
async fn should_apply_in_place_memory_reductions_and_raises_without_restarting_the_worker() {
    let demotion_context =
        DemotionTestContext::launch(IN_PLACE_TRUSTED_GENERATION, TRUSTED_MODEL_ID).await;
    demotion_context.load_selected_model(TRUSTED_MODEL_ID).await;
    let trusted_process_identifier = recorded_worker_identifier(
        &demotion_context.temporary_directory,
        IN_PLACE_TRUSTED_GENERATION,
    );

    for (updated_ceiling_bytes, updated_generation) in [
        (
            DEMOTION_CEILING_BYTES - 1_000_000_000,
            IN_PLACE_REDUCED_GENERATION,
        ),
        (
            DEMOTION_CEILING_BYTES + 1_000_000_000,
            IN_PLACE_RAISED_GENERATION,
        ),
    ] {
        demotion_context
            .worker_handle
            .stage_memory_configuration_generation(updated_generation.to_owned());
        let update_outcome = timeout(
            JOURNEY_TIMEOUT,
            demotion_context
                .worker_handle
                .update_mlx_memory_limit(updated_ceiling_bytes, updated_generation.to_owned()),
        )
        .await
        .expect("the in-place update should answer inside the journey budget")
        .expect("the fixture worker should adopt an in-place ceiling");
        demotion_context
            .worker_handle
            .record_memory_configuration_generation(updated_generation.to_owned(), update_outcome);
        assert_eq!(
            update_outcome,
            MlxMemoryLimitUpdateOutcome::Applied,
            "an in-place ceiling change must be adopted without a restart"
        );
    }

    let updated_health = demotion_context.worker_handle.worker_health_snapshot();
    assert_eq!(updated_health.status, WorkerHealthStatus::Ready);
    assert_eq!(
        updated_health.ready_model_id.as_deref(),
        Some(TRUSTED_MODEL_ID),
        "an in-place ceiling change must not drop the loaded model"
    );
    assert_eq!(
        updated_health
            .worker_runtime_feature_configuration
            .as_ref()
            .map(|runtime_configuration| runtime_configuration.configuration_generation.as_str()),
        Some(IN_PLACE_RAISED_GENERATION),
        "the last applied generation must own the live memory policy"
    );
    for restarted_generation in [IN_PLACE_REDUCED_GENERATION, IN_PLACE_RAISED_GENERATION] {
        assert!(
            !demotion_context
                .temporary_directory
                .join(format!(
                    "{WORKER_PROCESS_IDENTIFIER_PREFIX}{restarted_generation}.pid"
                ))
                .exists(),
            "an in-place update must never launch a worker process for {restarted_generation}"
        );
    }
    let launch_order_log = std::fs::read_to_string(
        demotion_context
            .temporary_directory
            .join(LAUNCH_ORDER_LOG_FILE),
    )
    .expect("the trusted worker should have recorded its launch observation");
    assert_eq!(
        launch_order_log.lines().count(),
        1,
        "an in-place update must never launch a worker, recorded {launch_order_log:?}"
    );
    assert_eq!(
        recorded_worker_identifier(
            &demotion_context.temporary_directory,
            IN_PLACE_TRUSTED_GENERATION,
        ),
        trusted_process_identifier,
        "the same worker process must own the whole in-place journey"
    );
    assert!(
        process_is_running(&trusted_process_identifier),
        "the worker must stay alive across in-place ceiling changes"
    );
    demotion_context.shutdown().await;
}
