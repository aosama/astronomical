//! Unit tests for the not-running disclosure wording: the channel name and
//! every probed socket must reach the user verbatim, because a failure that
//! names its channel is the only way to expose a wrong-channel binding
//! (issue #1392).

use std::path::PathBuf;

use astronomical_config::AstronomicalRuntimeInstance;

use crate::daemon_probe::daemon_not_running_disclosure;

#[test]
fn should_name_the_channel_and_the_single_probed_socket() {
    let probed_socket_paths = [PathBuf::from("/Users/shared/.astronomical-dev/daemon.sock")];
    let disclosure = daemon_not_running_disclosure(
        AstronomicalRuntimeInstance::Development,
        &probed_socket_paths,
    );
    assert_eq!(
        disclosure,
        "Astronomical (development channel) isn't running — start it, then retry. (probed socket: /Users/shared/.astronomical-dev/daemon.sock)"
    );
}

#[test]
fn should_list_every_probed_socket_and_pluralize_when_several_exist() {
    let probed_socket_paths = [
        PathBuf::from("/tmp/first-astronomical.sock"),
        PathBuf::from("/tmp/second-astronomical.sock"),
    ];
    let disclosure =
        daemon_not_running_disclosure(AstronomicalRuntimeInstance::Stable, &probed_socket_paths);
    assert_eq!(
        disclosure,
        "Astronomical (stable channel) isn't running — start it, then retry. (probed sockets: /tmp/first-astronomical.sock, /tmp/second-astronomical.sock)"
    );
}

#[test]
fn should_report_none_when_no_socket_path_could_be_resolved() {
    let disclosure = daemon_not_running_disclosure(AstronomicalRuntimeInstance::Stable, &[]);
    assert_eq!(
        disclosure,
        "Astronomical (stable channel) isn't running — start it, then retry. (probed sockets: none)"
    );
}
