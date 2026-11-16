//! Unit tests for making a calibrated model visible to the daemon.

use std::net::TcpListener;
use std::time::Duration;

use astronomical_config::AstronomicalRuntimeInstance;

use super::*;

fn models_root() -> Vec<PathBuf> {
    vec![PathBuf::from("models")]
}

#[test]
fn should_say_the_model_is_served_now_when_the_daemon_lists_it() {
    let written_text = written_folder_text(
        Path::new("models/calibrated/laya-verona"),
        "laya-verona",
        NewModelVisibility::Listed,
        &models_root(),
    );
    assert!(written_text.contains("Wrote the new model to models/calibrated/laya-verona."));
    assert!(written_text.contains("Astronomical serves it now; ask for it as laya-verona."));
}

#[test]
fn should_say_when_the_model_appears_if_the_daemon_was_busy() {
    let written_text = written_folder_text(
        Path::new("models/laya-verona"),
        "laya-verona",
        NewModelVisibility::ReloadDeclined,
        &models_root(),
    );
    assert!(written_text.contains("was busy"));
    assert!(written_text.contains("once Astronomical restarts"));
    assert!(!written_text.contains("serves it now"));
}

#[test]
fn should_say_where_to_move_a_folder_the_daemon_does_not_look_in() {
    let written_text = written_folder_text(
        Path::new("elsewhere/laya-verona"),
        "laya-verona",
        NewModelVisibility::NotFound,
        &models_root(),
    );
    assert!(written_text.contains(
        "Astronomical does not look for models in elsewhere. Move the folder into models,"
    ));
}

#[tokio::test]
async fn should_treat_an_unreachable_daemon_as_a_reload_for_later() {
    // A port that was free a moment ago: nothing answers there.
    let closed_address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: Vec::new(),
        request_timeout: Duration::from_millis(500),
        runtime_instance: AstronomicalRuntimeInstance::Development,
    };
    let visibility = make_new_model_visible(closed_address, &daemon_probe, "laya-verona").await;
    assert_eq!(visibility, NewModelVisibility::ReloadDeclined);
}
