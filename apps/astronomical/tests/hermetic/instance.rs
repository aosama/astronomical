use std::path::Path;

use astronomical_config::AstronomicalRuntimeInstance;

#[test]
fn should_treat_the_stable_app_bundle_as_stable_loopback() {
    let executable_path = Path::new("/Applications/Astronomical.app/Contents/MacOS/astronomical");
    assert_eq!(
        astronomical_cli::runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Stable
    );
}

#[test]
fn should_treat_the_development_app_bundle_as_development_loopback() {
    let executable_path =
        Path::new("/tmp/Astronomical Development.app/Contents/MacOS/astronomical");
    assert_eq!(
        astronomical_cli::runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Development
    );
}

#[test]
fn should_treat_unpackaged_binaries_as_development() {
    let executable_path = Path::new("/tmp/target/debug/astronomical");
    assert_eq!(
        astronomical_cli::runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Development
    );
}

#[test]
fn should_resolve_a_symlink_to_the_bundle_it_points_into() {
    let sandbox_directory =
        std::env::temp_dir().join(format!("astronomical-cli-instance-{}", std::process::id()));
    let bundle_executable_directory = sandbox_directory.join("Astronomical.app/Contents/MacOS");
    std::fs::create_dir_all(&bundle_executable_directory)
        .expect("bundle directory should be creatable");
    let bundle_executable = bundle_executable_directory.join("astronomical");
    std::fs::write(&bundle_executable, b"").expect("bundle executable should be writable");
    let installed_link = sandbox_directory.join("bin").join("astronomical");
    std::fs::create_dir_all(installed_link.parent().expect("parent exists"))
        .expect("bin directory should be creatable");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&bundle_executable, &installed_link)
        .expect("symlink should be creatable");

    assert_eq!(
        astronomical_cli::runtime_instance_from_executable_path(&installed_link),
        AstronomicalRuntimeInstance::Stable
    );

    std::fs::remove_dir_all(&sandbox_directory).expect("sandbox should be removable");
}

#[test]
fn should_keep_falling_back_to_the_given_path_when_it_cannot_be_canonicalized() {
    let executable_path = Path::new("/nonexistent/astronomical");
    assert_eq!(
        astronomical_cli::runtime_instance_from_executable_path(executable_path),
        AstronomicalRuntimeInstance::Development
    );
}

#[test]
fn should_try_the_binarys_own_instance_before_the_other_channel() {
    assert_eq!(
        astronomical_cli::candidate_instances(AstronomicalRuntimeInstance::Stable),
        [
            AstronomicalRuntimeInstance::Stable,
            AstronomicalRuntimeInstance::Development
        ]
    );
    assert_eq!(
        astronomical_cli::candidate_instances(AstronomicalRuntimeInstance::Development),
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
