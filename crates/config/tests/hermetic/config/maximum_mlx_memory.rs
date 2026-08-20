use super::*;

#[test]
fn should_leave_maximum_mlx_memory_unset_when_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");

    assert_eq!(
        astronomical_config
            .maximum_mlx_memory_bytes()
            .expect("memory ceiling should resolve"),
        None
    );
}

#[test]
fn should_convert_maximum_mlx_memory_decimal_gigabytes_to_bytes() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"maximum_mlx_memory_gb": 32}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .maximum_mlx_memory_bytes()
            .expect("memory ceiling should resolve"),
        Some(32_000_000_000)
    );
}

#[test]
fn should_reject_zero_maximum_mlx_memory_gigabytes() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"maximum_mlx_memory_gb": 0}"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::InvalidMaximumMlxMemoryGb { .. })
    ));
}

#[test]
fn should_reject_maximum_mlx_memory_byte_overflow() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        &format!(r#"{{"maximum_mlx_memory_gb": {}}}"#, u64::MAX),
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::InvalidMaximumMlxMemoryGb { .. })
    ));
}

#[test]
fn should_atomically_update_maximum_mlx_memory_without_losing_other_config_fields() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let original_config_bytes = br#"{
      "model_directories": ["/models/astronomical"],
      "mtp_enabled": true,
      "chunking": {}
    }"#;
    write_config(
        temporary_home_directory.path(),
        std::str::from_utf8(original_config_bytes).expect("fixture should be UTF-8"),
    );

    let config_update = write_maximum_mlx_memory_gb(
        temporary_home_directory.path().join(".astronomical"),
        Some(32),
    )
    .expect("memory ceiling should be persisted");

    assert_eq!(
        config_update.prior_config_bytes,
        Some(original_config_bytes.to_vec())
    );
    assert_eq!(
        std::fs::read(
            temporary_home_directory
                .path()
                .join(".astronomical/config.legacy-v0.json")
        )
        .expect("legacy backup should be readable"),
        original_config_bytes
    );
    let persisted_config_bytes = std::fs::read(
        temporary_home_directory
            .path()
            .join(".astronomical")
            .join("config.json"),
    )
    .expect("persisted config should be readable");
    let persisted_config: serde_json::Value =
        serde_json::from_slice(&persisted_config_bytes).expect("persisted config should be JSON");

    assert_eq!(
        persisted_config
            .get("runtime")
            .and_then(|runtime| runtime.get("model_directories"))
            .and_then(serde_json::Value::as_array),
        Some(&vec![serde_json::Value::String(
            "/models/astronomical".to_owned()
        )])
    );
    assert_eq!(
        persisted_config.get("schema_version"),
        Some(&serde_json::json!(1))
    );
    assert_eq!(
        persisted_config
            .get("runtime")
            .and_then(|runtime| runtime.get("maximum_mlx_memory_gb"))
            .and_then(serde_json::Value::as_u64),
        Some(32)
    );
}

#[test]
fn should_remove_maximum_mlx_memory_override_when_reset_to_automatic() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let original_config_bytes =
        br#"{"maximum_mlx_memory_gb": 32, "mtp_enabled": true, "chunking": {}}"#;
    write_config(
        temporary_home_directory.path(),
        std::str::from_utf8(original_config_bytes).expect("fixture should be UTF-8"),
    );

    let config_update =
        write_maximum_mlx_memory_gb(temporary_home_directory.path().join(".astronomical"), None)
            .expect("memory ceiling override should be removed");

    assert_eq!(
        config_update.prior_config_bytes,
        Some(original_config_bytes.to_vec())
    );
    assert_eq!(
        std::fs::read(
            temporary_home_directory
                .path()
                .join(".astronomical/config.legacy-v0.json")
        )
        .expect("legacy backup should be readable"),
        original_config_bytes
    );
    let persisted_config_bytes = std::fs::read(
        temporary_home_directory
            .path()
            .join(".astronomical")
            .join("config.json"),
    )
    .expect("persisted config should be readable");
    let persisted_config: serde_json::Value =
        serde_json::from_slice(&persisted_config_bytes).expect("persisted config should be JSON");

    assert!(
        persisted_config["runtime"]
            .get("maximum_mlx_memory_gb")
            .is_none()
    );
    assert_eq!(
        persisted_config.get("schema_version"),
        Some(&serde_json::json!(1))
    );
}

#[test]
fn should_leave_original_config_unchanged_when_candidate_validation_fails() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let original_config_bytes = br#"{"unexpected_field": true, "chunking": {}}"#;
    write_config(
        temporary_home_directory.path(),
        std::str::from_utf8(original_config_bytes).expect("fixture should be UTF-8"),
    );

    let persist_result = write_maximum_mlx_memory_gb(
        temporary_home_directory.path().join(".astronomical"),
        Some(32),
    );

    assert!(matches!(
        persist_result,
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
    assert_eq!(
        std::fs::read(
            temporary_home_directory
                .path()
                .join(".astronomical")
                .join("config.json"),
        )
        .expect("original config should remain readable"),
        original_config_bytes
    );
}
