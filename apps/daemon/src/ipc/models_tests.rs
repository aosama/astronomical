//! Unit tests for the pure `DownloadStart` decision: every state the daemon's
//! existing job can be in, for the same model and for a different one.

use super::{DownloadStartAction, download_start_action, paused_refusal_reason, running_state_of};
use crate::library::DownloadJobState;

const MODEL: &str = "black-forest-labs/FLUX.2-klein-4B";
const OTHER_MODEL: &str = "mlx-community/Qwen-Image-2.1-MLX-4bit";

const EVERY_STATE: [DownloadJobState; 7] = [
    DownloadJobState::CheckingDisk,
    DownloadJobState::FetchingManifest,
    DownloadJobState::Downloading,
    DownloadJobState::Paused,
    DownloadJobState::Verifying,
    DownloadJobState::Publishing,
    DownloadJobState::Failed,
];

#[test]
fn should_start_a_fresh_download_when_the_daemon_holds_no_job() {
    assert_eq!(
        download_start_action(None, MODEL),
        DownloadStartAction::Start
    );
}

#[test]
fn should_resume_the_same_models_paused_or_failed_job() {
    for resumable_state in [DownloadJobState::Paused, DownloadJobState::Failed] {
        assert_eq!(
            download_start_action(Some((MODEL, resumable_state)), MODEL),
            DownloadStartAction::Resume,
            "{resumable_state:?}"
        );
    }
}

/// The rerun-after-timeout case: the previous command left the download
/// running, so asking again must attach to it, not be refused as busy.
#[test]
fn should_attach_to_the_same_models_download_in_every_running_state() {
    for running_state in [
        DownloadJobState::CheckingDisk,
        DownloadJobState::FetchingManifest,
        DownloadJobState::Downloading,
        DownloadJobState::Verifying,
        DownloadJobState::Publishing,
    ] {
        assert_eq!(
            download_start_action(Some((MODEL, running_state)), MODEL),
            DownloadStartAction::AlreadyRunning,
            "{running_state:?}"
        );
    }
}

#[test]
fn should_refuse_a_different_model_that_is_running_or_paused_and_name_the_holder() {
    for existing_state in EVERY_STATE {
        if existing_state == DownloadJobState::Failed {
            continue;
        }
        assert_eq!(
            download_start_action(Some((OTHER_MODEL, existing_state)), MODEL),
            DownloadStartAction::BusyWith {
                other_huggingface_id: OTHER_MODEL.to_owned(),
                is_paused: existing_state == DownloadJobState::Paused,
            },
            "{existing_state:?}"
        );
    }
}

/// A failed job of another model is not running, and the CLI had no way to
/// clear it, so it blocked every later download from the CLI (issue #1897).
#[test]
fn should_start_over_a_different_models_failed_download() {
    assert_eq!(
        download_start_action(Some((OTHER_MODEL, DownloadJobState::Failed)), MODEL),
        DownloadStartAction::Start
    );
}

#[test]
fn should_still_resume_the_same_models_failed_download() {
    assert_eq!(
        download_start_action(Some((MODEL, DownloadJobState::Failed)), MODEL),
        DownloadStartAction::Resume
    );
}

#[test]
fn should_decide_every_state_for_the_same_model_without_a_gap() {
    for existing_state in EVERY_STATE {
        let action = download_start_action(Some((MODEL, existing_state)), MODEL);

        assert!(
            matches!(
                action,
                DownloadStartAction::Resume | DownloadStartAction::AlreadyRunning
            ),
            "{existing_state:?} -> {action:?}"
        );
    }
}

#[test]
fn should_treat_model_ids_that_differ_only_in_case_as_different_models() {
    assert_eq!(
        download_start_action(
            Some((
                "Black-Forest-Labs/FLUX.2-klein-4B",
                DownloadJobState::Downloading
            )),
            MODEL
        ),
        DownloadStartAction::BusyWith {
            other_huggingface_id: "Black-Forest-Labs/FLUX.2-klein-4B".to_owned(),
            is_paused: false,
        }
    );
}

#[test]
fn should_point_the_cli_at_the_models_page_and_not_ask_it_to_wait_on_a_paused_download() {
    let paused_reason = paused_refusal_reason(OTHER_MODEL, MODEL);

    assert!(paused_reason.contains("is paused; resume or cancel it on Observatory's Models page"));
    assert!(!paused_reason.contains("wait"));
    assert!(paused_reason.contains(OTHER_MODEL) && paused_reason.contains(MODEL));
}

#[test]
fn should_count_a_paused_job_as_downloading_while_its_task_runs() {
    assert_eq!(
        running_state_of(DownloadJobState::Paused, true),
        DownloadJobState::Downloading
    );
}

#[test]
fn should_keep_a_paused_job_paused_when_no_task_runs() {
    assert_eq!(
        running_state_of(DownloadJobState::Paused, false),
        DownloadJobState::Paused
    );
}

#[test]
fn should_leave_every_other_state_as_saved_whether_or_not_a_task_runs() {
    for saved_state in EVERY_STATE
        .into_iter()
        .filter(|saved_state| *saved_state != DownloadJobState::Paused)
    {
        assert_eq!(running_state_of(saved_state, true), saved_state);
        assert_eq!(running_state_of(saved_state, false), saved_state);
    }
}
