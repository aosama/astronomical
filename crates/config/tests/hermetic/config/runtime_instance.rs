use super::*;
use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};

#[test]
fn should_keep_stable_and_development_state_and_endpoints_separate() {
    let fictional_home_directory = PathBuf::from("/Users/example");

    let stable_paths = AstronomicalInstancePaths::for_home_directory(
        &fictional_home_directory,
        AstronomicalRuntimeInstance::Stable,
    );
    let development_paths = AstronomicalInstancePaths::for_home_directory(
        &fictional_home_directory,
        AstronomicalRuntimeInstance::Development,
    );

    assert_eq!(
        stable_paths.state_directory(),
        fictional_home_directory.join(".astronomical")
    );
    assert_eq!(
        development_paths.state_directory(),
        fictional_home_directory.join(".astronomical-dev")
    );
    assert_eq!(
        stable_paths.default_bind_address().to_string(),
        "127.0.0.1:6732"
    );
    assert_eq!(
        development_paths.default_bind_address().to_string(),
        "127.0.0.1:6733"
    );
    assert_ne!(
        stable_paths.config_file_path(),
        development_paths.config_file_path()
    );
    assert_ne!(
        stable_paths.prompt_cache_directory(),
        development_paths.prompt_cache_directory()
    );
    assert_eq!(
        stable_paths.models_directory(),
        fictional_home_directory.join(".astronomical/models")
    );
    assert_eq!(
        development_paths.models_directory(),
        fictional_home_directory.join(".astronomical-dev/models")
    );
    assert_ne!(
        stable_paths.models_directory(),
        development_paths.models_directory()
    );
    assert_ne!(
        stable_paths.logging_directory(),
        development_paths.logging_directory()
    );
    assert_ne!(
        stable_paths.instance_lock_file_path(),
        development_paths.instance_lock_file_path()
    );
    assert!(stable_paths.is_standard_state_directory());
    assert!(development_paths.is_standard_state_directory());
}

#[test]
fn should_keep_every_writable_path_beneath_an_explicit_test_state_directory() {
    let test_state_directory = tempfile::tempdir().expect("test state directory should be created");
    let instance_paths = AstronomicalInstancePaths::for_explicit_state_directory(
        test_state_directory.path().to_path_buf(),
        "127.0.0.1:0"
            .parse()
            .expect("test bind address should parse"),
    );
    assert!(!instance_paths.is_standard_state_directory());
    assert_eq!(
        instance_paths.models_directory(),
        test_state_directory.path().join("models")
    );

    for writable_path in [
        instance_paths.config_file_path(),
        instance_paths.prompt_cache_directory(),
        instance_paths.models_directory(),
        instance_paths.logging_directory(),
        instance_paths.daemon_ownership_file_path(),
        instance_paths.instance_lock_file_path(),
        instance_paths.qwen_thinking_channel_seed_file_path(),
    ] {
        assert!(writable_path.starts_with(test_state_directory.path()));
    }
}

#[test]
fn should_generate_the_development_first_run_config_with_the_development_port() {
    let fictional_home_directory = tempfile::tempdir().expect("fictional home should be created");
    let development_paths = AstronomicalInstancePaths::for_home_directory(
        fictional_home_directory.path(),
        AstronomicalRuntimeInstance::Development,
    );

    let development_config =
        AstronomicalConfig::load_from_instance_paths(development_paths.clone())
            .expect("development config should be created");

    assert_eq!(
        development_config
            .supervisor_bind_address()
            .expect("development address should resolve")
            .to_string(),
        "127.0.0.1:6733"
    );
    assert!(development_paths.config_file_path().is_file());
}

#[test]
fn should_load_a_supplied_test_home_only_from_the_development_channel() {
    let test_home_directory = tempfile::tempdir().expect("test home should be created");
    let stable_state_directory = test_home_directory.path().join(".astronomical");
    let development_state_directory = test_home_directory.path().join(".astronomical-dev");
    std::fs::create_dir_all(&stable_state_directory).expect("Stable fixture should be created");
    std::fs::create_dir_all(&development_state_directory)
        .expect("Development fixture should be created");
    std::fs::write(
        stable_state_directory.join("config.json"),
        b"not valid JSON",
    )
    .expect("Stable sentinel should be written");
    std::fs::write(
        development_state_directory.join("config.json"),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}}"#,
    )
    .expect("Development config should be written");

    let development_config =
        AstronomicalConfig::load_from_development_home_directory(test_home_directory.path())
            .expect("Development config must load without reading the Stable sentinel");

    assert_eq!(
        development_config
            .supervisor_bind_address()
            .expect("Development endpoint should resolve")
            .to_string(),
        "127.0.0.1:6733"
    );
    assert_eq!(
        std::fs::read(stable_state_directory.join("config.json"))
            .expect("Stable sentinel should remain readable"),
        b"not valid JSON"
    );
}

#[test]
fn should_allow_an_explicit_test_state_directory_to_select_an_ephemeral_endpoint() {
    let test_state_directory = tempfile::tempdir().expect("test state directory should be created");
    let explicit_paths = AstronomicalInstancePaths::for_explicit_state_directory(
        test_state_directory.path().to_path_buf(),
        "127.0.0.1:0"
            .parse()
            .expect("test bind address should parse"),
    );
    std::fs::write(
        explicit_paths.config_file_path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}}"#,
    )
        .expect("explicit config should be written");

    let explicit_config = AstronomicalConfig::load_from_instance_paths(explicit_paths)
        .expect("explicit config should load");

    assert_eq!(
        explicit_config
            .supervisor_bind_address()
            .expect("explicit endpoint should resolve")
            .to_string(),
        "127.0.0.1:0"
    );
}

#[test]
fn should_assign_an_ephemeral_endpoint_to_a_custom_channel_state_directory() {
    let test_state_directory = tempfile::tempdir().expect("test state directory should be created");
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        test_state_directory.path().to_path_buf(),
        AstronomicalRuntimeInstance::Development,
    );

    assert_eq!(
        instance_paths.default_bind_address().to_string(),
        "127.0.0.1:0"
    );
    assert!(!instance_paths.is_standard_state_directory());
}

#[test]
fn should_canonicalize_a_valid_user_home_before_deriving_standard_state() {
    let user_home_directory = tempfile::tempdir().expect("user home should be created");

    let development_paths = AstronomicalInstancePaths::for_user_home_directory(
        user_home_directory.path(),
        AstronomicalRuntimeInstance::Development,
    )
    .expect("valid user home should resolve");

    assert_eq!(
        development_paths.state_directory(),
        user_home_directory
            .path()
            .canonicalize()
            .expect("user home should canonicalize")
            .join(".astronomical-dev")
    );
}

#[test]
fn should_reject_a_relative_user_home_before_deriving_standard_state() {
    let home_directory_error = AstronomicalInstancePaths::for_user_home_directory(
        PathBuf::from("relative-home"),
        AstronomicalRuntimeInstance::Stable,
    )
    .expect_err("relative HOME must be rejected");

    assert!(matches!(
        home_directory_error,
        AstronomicalConfigError::PathMustBeAbsolute { field_name, .. } if field_name == "HOME"
    ));
}

#[test]
fn should_reject_the_filesystem_root_as_a_user_home() {
    let home_directory_error = AstronomicalInstancePaths::for_user_home_directory(
        PathBuf::from("/"),
        AstronomicalRuntimeInstance::Stable,
    )
    .expect_err("filesystem root must not become Astronomical user state");

    assert!(matches!(
        home_directory_error,
        AstronomicalConfigError::HomeDirectoryMustNotBeRoot
    ));
}

#[test]
fn should_resolve_thinking_markdown_under_the_instance_state_directory() {
    let fictional_home_directory = PathBuf::from("/Users/example");
    let development_paths = AstronomicalInstancePaths::for_home_directory(
        &fictional_home_directory,
        AstronomicalRuntimeInstance::Development,
    );
    let stable_paths = AstronomicalInstancePaths::for_home_directory(
        &fictional_home_directory,
        AstronomicalRuntimeInstance::Stable,
    );

    assert_eq!(
        development_paths.qwen_thinking_channel_seed_file_path(),
        fictional_home_directory.join(".astronomical-dev/thinking.md")
    );
    assert_eq!(
        stable_paths.qwen_thinking_channel_seed_file_path(),
        fictional_home_directory.join(".astronomical/thinking.md")
    );
}

#[test]
fn should_resolve_app_store_state_beneath_the_application_support_directory() {
    let fictional_application_support_directory =
        PathBuf::from("/Users/example/Library/Application Support");

    let stable_paths = AstronomicalInstancePaths::for_application_support_directory(
        &fictional_application_support_directory,
        AstronomicalRuntimeInstance::Stable,
    );

    assert_eq!(
        stable_paths.state_directory(),
        fictional_application_support_directory.join("Astronomical")
    );
    assert_eq!(
        stable_paths.models_directory(),
        fictional_application_support_directory.join("Astronomical/models")
    );
    assert_eq!(
        stable_paths.prompt_cache_directory(),
        fictional_application_support_directory.join("Astronomical/cache")
    );
    assert_eq!(
        stable_paths.logging_directory(),
        fictional_application_support_directory.join("Astronomical/logs")
    );
    assert_eq!(
        stable_paths.config_file_path(),
        fictional_application_support_directory.join("Astronomical/config.json")
    );
    // Standard-instance endpoint guards must carry over so the store build
    // keeps the same loopback discipline as the direct channel.
    assert!(stable_paths.is_standard_state_directory());
    assert_eq!(
        stable_paths.default_bind_address().to_string(),
        "127.0.0.1:6732"
    );
}

#[test]
fn should_keep_app_store_stable_and_development_state_separate() {
    let fictional_application_support_directory =
        PathBuf::from("/Users/example/Library/Application Support");

    let stable_paths = AstronomicalInstancePaths::for_application_support_directory(
        &fictional_application_support_directory,
        AstronomicalRuntimeInstance::Stable,
    );
    let development_paths = AstronomicalInstancePaths::for_application_support_directory(
        &fictional_application_support_directory,
        AstronomicalRuntimeInstance::Development,
    );

    assert_eq!(
        stable_paths.state_directory(),
        fictional_application_support_directory.join("Astronomical")
    );
    assert_eq!(
        development_paths.state_directory(),
        fictional_application_support_directory.join("Astronomical Development")
    );
    assert_ne!(
        stable_paths.state_directory(),
        development_paths.state_directory()
    );
    assert_ne!(
        stable_paths.default_bind_address(),
        development_paths.default_bind_address()
    );
}

#[test]
fn should_never_share_app_store_state_with_the_home_dot_folder_channel() {
    let fictional_home_directory = PathBuf::from("/Users/example");
    let fictional_application_support_directory =
        PathBuf::from("/Users/example/Library/Application Support");

    let direct_paths = AstronomicalInstancePaths::for_home_directory(
        &fictional_home_directory,
        AstronomicalRuntimeInstance::Stable,
    );
    let app_store_paths = AstronomicalInstancePaths::for_application_support_directory(
        &fictional_application_support_directory,
        AstronomicalRuntimeInstance::Stable,
    );

    assert_ne!(
        direct_paths.state_directory(),
        app_store_paths.state_directory()
    );
}
