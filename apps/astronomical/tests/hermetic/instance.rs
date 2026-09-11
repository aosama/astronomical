use std::path::Path;

use astronomical_cli::{candidate_instances, runtime_instance_from_executable_path};
use astronomical_config::AstronomicalRuntimeInstance;

#[test]
fn should_treat_the_stable_app_bundle_as_stable_loopback() {
    let executable_path = Path::new("/Applications/Astronomical.app/Contents/MacOS/astronomical");
    assert_eq!(
        runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Stable
    );
}

#[test]
fn should_treat_the_development_app_bundle_as_development_loopback() {
    let executable_path =
        Path::new("/tmp/Astronomical Development.app/Contents/MacOS/astronomical");
    assert_eq!(
        runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Development
    );
}

#[test]
fn should_treat_unpackaged_binaries_as_development() {
    let executable_path = Path::new("/tmp/target/debug/astronomical");
    assert_eq!(
        runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Development
    );
}

#[test]
fn should_try_the_binarys_own_instance_before_the_other_channel() {
    assert_eq!(
        candidate_instances(AstronomicalRuntimeInstance::Stable),
        [
            AstronomicalRuntimeInstance::Stable,
            AstronomicalRuntimeInstance::Development
        ]
    );
    assert_eq!(
        candidate_instances(AstronomicalRuntimeInstance::Development),
        [
            AstronomicalRuntimeInstance::Development,
            AstronomicalRuntimeInstance::Stable
        ]
    );
}

#[test]
fn should_expose_stable_and_development_loopback_ports() {
    assert_eq!(
        AstronomicalRuntimeInstance::Stable
            .loopback_socket_addr()
            .to_string(),
        "127.0.0.1:6732"
    );
    assert_eq!(
        AstronomicalRuntimeInstance::Development
            .loopback_socket_addr()
            .to_string(),
        "127.0.0.1:6733"
    );
}
