//! Acceptance coverage for one-way atomic legacy configuration migration.

use std::fs;
use std::os::unix::fs::PermissionsExt;

use super::*;

#[test]
fn should_migrate_representable_legacy_configuration_to_v1() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let model_root = temporary_home_directory.path().join("models");
    let model_directory = model_root.join("target");
    fs::create_dir_all(&model_directory).expect("model directory should be created");
    super::super::model_discovery::write_minimal_model_config(
        &model_directory,
        "qwen3_5_moe",
        65_536,
    );
    super::super::model_discovery::write_required_model_files(&model_directory);
    let legacy_config_json = serde_json::json!({
        "model_directories": [model_root],
        "max_output_tokens": 4096,
        "maximum_mlx_memory_gb": 16,
        "persistent_prompt_cache_enabled": false,
        "prompt_cache_max_size_gb": 20,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "mtp_draft_depth": 2,
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 4096,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 1024,
            "full_attention_key_value_growth_tokens": 512,
            "prefill_graph_submission_layer_interval": 2,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 4,
            "prompt_cache_block_tokens": 512,
            "prompt_cache_common_prefix_stride_blocks": 8
        },
        "logging": {"level": "info", "retained_files": 4}
    })
    .to_string();
    write_config(temporary_home_directory.path(), &legacy_config_json);
    let config_directory = temporary_home_directory.path().join(".astronomical");
    let original_config_bytes = fs::read(config_directory.join("config.json"))
        .expect("legacy config should be readable before migration");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("representable legacy config should migrate");
    let migrated_bytes = fs::read(
        temporary_home_directory
            .path()
            .join(".astronomical/config.json"),
    )
    .expect("migrated config should be readable");
    let migrated_json: serde_json::Value =
        serde_json::from_slice(&migrated_bytes).expect("migrated config should be JSON");
    let model_config = astronomical_config
        .resolved_model_config("target", 65_536)
        .expect("migrated model policy should resolve");

    assert_eq!(migrated_json["schema_version"], 1);
    let legacy_backup_path = config_directory.join("config.legacy-v0.json");
    assert_eq!(
        fs::read(&legacy_backup_path).expect("legacy backup should be readable"),
        original_config_bytes
    );
    assert_eq!(
        fs::metadata(legacy_backup_path)
            .expect("legacy backup metadata should be readable")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(migrated_json["runtime"]["maximum_mlx_memory_gb"], 16);
    assert_eq!(migrated_json["prompt_cache"]["enabled"], false);
    assert!(migrated_json.get("max_output_tokens").is_none());
    assert_eq!(model_config.maximum_output_tokens(), 4_096);
    assert!(!model_config.mtp_enabled());
    assert_eq!(model_config.mtp_draft_depth(), Some(2));
    assert_eq!(
        model_config
            .chunking()
            .fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        1_024
    );
    assert_eq!(
        model_config.chunking().prompt_cache_block_tokens(),
        Some(512)
    );
    assert_eq!(
        model_config
            .chunking()
            .prompt_cache_common_prefix_stride_blocks(),
        8
    );
}

#[test]
fn should_strip_retired_speculative_prefill_settings_while_migrating_legacy_config() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let legacy_config_json = serde_json::json!({
        "model_directories": [],
        "maximum_mlx_memory_gb": 16,
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2048,
            "speculative_prefill_draft_forward_tokens": 1536
        },
        "speculative_prefill": {
            "enabled": true,
            "target_model_id": "target",
            "draft_model_id": "draft",
            "keep_percentage": 20
        }
    })
    .to_string();
    write_config(temporary_home_directory.path(), &legacy_config_json);
    let config_file_path = temporary_home_directory
        .path()
        .join(".astronomical/config.json");

    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("retired legacy SpeculativePrefill settings should not block config migration");

    let migrated_json: serde_json::Value = serde_json::from_slice(
        &fs::read(&config_file_path).expect("migrated config should be readable"),
    )
    .expect("migrated config should be JSON");
    let retained_legacy_backup = fs::read(config_file_path.with_file_name("config.legacy-v0.json"))
        .expect("legacy migration should retain the original config bytes");

    assert_eq!(migrated_json["schema_version"], 1);
    assert_eq!(retained_legacy_backup, legacy_config_json.as_bytes());
    assert!(
        migrated_json["chunking"]
            .get("speculative_prefill_draft_forward_tokens")
            .is_none()
    );
    assert!(
        migrated_json["models"]
            .as_object()
            .is_none_or(|models| models.values().all(|model| {
                model
                    .get("acceleration")
                    .and_then(|acceleration| acceleration.get("speculative_prefill"))
                    .is_none()
            }))
    );
}

#[test]
fn should_strip_retired_speculative_prefill_settings_from_v1_config() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema":"./astronomical-config.schema.json",
          "schema_version":1,
          "runtime":{"model_directories":[]},
          "chunking":{"speculative_prefill_draft_forward_tokens":1536},
          "models":{
            "organization/target":{
              "chunking":{"speculative_prefill_draft_forward_tokens":1024},
              "acceleration":{
                "speculative_prefill":{"draft_model_id":"organization/draft"},
                "mtp":{"enabled":true,"draft_depth":2}
              }
            }
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("retired SpeculativePrefill fields should be stripped before strict parsing");
    let resolved_model = astronomical_config
        .resolved_model_config("organization/target", 65_536)
        .expect("the remaining MTP policy should resolve");
    let persisted_json: serde_json::Value = serde_json::from_slice(
        &fs::read(
            temporary_home_directory
                .path()
                .join(".astronomical/config.json"),
        )
        .expect("sanitized config should be readable"),
    )
    .expect("sanitized config should be JSON");

    assert!(resolved_model.mtp_enabled());
    assert_eq!(resolved_model.mtp_draft_depth(), Some(2));
    assert!(
        persisted_json["chunking"]
            .get("speculative_prefill_draft_forward_tokens")
            .is_none()
    );
    assert!(
        persisted_json["models"]["organization/target"]["chunking"]
            .get("speculative_prefill_draft_forward_tokens")
            .is_none()
    );
    assert!(
        persisted_json["models"]["organization/target"]["acceleration"]
            .get("speculative_prefill")
            .is_none()
    );
    assert_eq!(
        persisted_json["models"]["organization/target"]["acceleration"]["mtp"]["draft_depth"],
        2
    );
}

#[test]
fn should_preserve_original_bytes_when_legacy_migration_cannot_preserve_behavior() {
    for legacy_config in [
        r#"{"model_directories":[],"max_output_tokens":4096}"#,
        r#"{"model_directories":[],"mtp_draft_depth":2}"#,
        r#"{"model_directories":[],"mtp_enabled":false}"#,
        r#"{"model_directories":[],"supervisor":{"bind_address":"127.0.0.1:12345"}}"#,
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(temporary_home_directory.path(), legacy_config);
        let config_path = temporary_home_directory
            .path()
            .join(".astronomical/config.json");
        let original_bytes = fs::read(&config_path).expect("legacy config should be readable");

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::LegacyMigration { .. })
        ));
        assert_eq!(
            fs::read(&config_path).expect("failed migration must retain config"),
            original_bytes
        );
        assert!(!config_path.with_file_name("config.legacy-v0.json").exists());
    }
}

#[test]
fn should_resume_migration_when_the_exact_legacy_backup_already_exists() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let legacy_config = r#"{"model_directories":[],"maximum_mlx_memory_gb":16}"#;
    write_config(temporary_home_directory.path(), legacy_config);
    let config_directory = temporary_home_directory.path().join(".astronomical");
    let legacy_backup_path = config_directory.join("config.legacy-v0.json");
    fs::write(&legacy_backup_path, legacy_config).expect("matching backup should be written");

    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("migration should resume with an exact existing backup");

    let migrated_json: serde_json::Value = serde_json::from_slice(
        &fs::read(config_directory.join("config.json"))
            .expect("migrated config should be readable"),
    )
    .expect("migrated config should be JSON");
    assert_eq!(migrated_json["schema_version"], 1);
    assert_eq!(
        fs::read(legacy_backup_path).expect("matching backup should remain readable"),
        legacy_config.as_bytes()
    );
}

#[test]
fn should_refuse_to_overwrite_a_different_legacy_backup() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let legacy_config = r#"{"model_directories":[],"maximum_mlx_memory_gb":16}"#;
    let conflicting_backup = br#"{"model_directories":[]}"#;
    write_config(temporary_home_directory.path(), legacy_config);
    let config_directory = temporary_home_directory.path().join(".astronomical");
    let config_path = config_directory.join("config.json");
    let legacy_backup_path = config_directory.join("config.legacy-v0.json");
    fs::write(&legacy_backup_path, conflicting_backup)
        .expect("conflicting backup should be written");

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::LegacyMigration { .. })
    ));
    assert_eq!(
        fs::read(config_path).expect("blocked migration must retain config"),
        legacy_config.as_bytes()
    );
    assert_eq!(
        fs::read(legacy_backup_path).expect("conflicting backup must remain readable"),
        conflicting_backup
    );
}

#[test]
fn should_not_create_a_legacy_backup_for_a_v1_configuration() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}}"#,
    );

    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("v1 config should load without migration");

    assert!(
        !temporary_home_directory
            .path()
            .join(".astronomical/config.legacy-v0.json")
            .exists()
    );
}
