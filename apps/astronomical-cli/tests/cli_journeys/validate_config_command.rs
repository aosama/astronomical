//! Integration acceptance journeys for `astronomical validate config` (issue #821
//! slice 1). The verb must load a configuration file in-process, report its
//! effective values, and fail with a recovery message on missing or invalid
//! documents. No daemon, no GPU, no network.

use std::path::PathBuf;

use astronomical_cli::{
    ValidateConfigArguments, ValidateConfigDependencies, ValidateConfigError, run_validate_config,
};
use astronomical_config::{AstronomicalInstancePaths, AstronomicalRuntimeInstance};

struct SandboxStateDirectory {
    _temporary_directory: tempfile::TempDir,
    instance_paths: AstronomicalInstancePaths,
}

fn sandbox_instance_paths() -> SandboxStateDirectory {
    let temporary_directory = tempfile::tempdir().expect("temporary state directory");
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        temporary_directory.path().join("state"),
        AstronomicalRuntimeInstance::Development,
    );
    SandboxStateDirectory {
        _temporary_directory: temporary_directory,
        instance_paths,
    }
}

const MINIMAL_VALID_CONFIG_JSON: &str = r#"{
  "$schema": "./astronomical-config.schema.json",
  "schema_version": 1,
  "runtime": { "model_directories": ["/absolute/model/library"] }
}"#;

fn write_config_file(instance_paths: &AstronomicalInstancePaths, contents: &str) -> PathBuf {
    let config_file_path = instance_paths.config_file_path();
    if let Some(parent_directory) = config_file_path.parent() {
        std::fs::create_dir_all(parent_directory).expect("state directory");
    }
    std::fs::write(&config_file_path, contents).expect("config file");
    config_file_path
}

fn rendered_report(
    instance_paths: &AstronomicalInstancePaths,
) -> Result<String, ValidateConfigError> {
    let mut rendered_output = Vec::new();
    let mut ignored_errors = Vec::new();
    run_validate_config(
        &ValidateConfigArguments {
            runtime_instance: AstronomicalRuntimeInstance::Development,
            render_json: false,
        },
        &mut ValidateConfigDependencies {
            instance_paths: instance_paths.clone(),
            stdout: &mut rendered_output,
            stderr: &mut ignored_errors,
        },
    )?;
    Ok(String::from_utf8(rendered_output).expect("utf-8 report"))
}

#[test]
fn should_report_effective_values_for_valid_config() {
    let sandbox_state = sandbox_instance_paths();
    write_config_file(&sandbox_state.instance_paths, MINIMAL_VALID_CONFIG_JSON);
    let rendered_text = rendered_report(&sandbox_state.instance_paths).expect("valid config");
    assert!(
        rendered_text.contains(
            sandbox_state
                .instance_paths
                .config_file_path()
                .to_string_lossy()
                .as_ref(),
        )
    );
    assert!(rendered_text.contains("Model directories: 1"));
    assert!(rendered_text.contains("/absolute/model/library"));
    assert!(rendered_text.contains("Runtime instance: development"));
    assert!(rendered_text.contains("Daemon bind address: 127.0.0.1"));
    assert!(rendered_text.ends_with('\n'));
}

#[test]
fn should_fail_when_config_file_is_missing() {
    let sandbox_state = sandbox_instance_paths();
    let validation_error =
        rendered_report(&sandbox_state.instance_paths).expect_err("no config file written");
    match validation_error {
        ValidateConfigError::ConfigFileMissing { config_file_path } => {
            assert_eq!(
                config_file_path,
                sandbox_state.instance_paths.config_file_path()
            );
        }
        other => panic!("expected missing config file error, got {other:?}"),
    }
}

#[test]
fn should_fail_when_config_document_is_invalid() {
    let sandbox_state = sandbox_instance_paths();
    write_config_file(&sandbox_state.instance_paths, r#"{ "unexpected": true }"#);
    let validation_error =
        rendered_report(&sandbox_state.instance_paths).expect_err("invalid config document");
    assert!(
        validation_error.to_string().contains(
            sandbox_state
                .instance_paths
                .config_file_path()
                .to_string_lossy()
                .as_ref(),
        )
    );
}

#[test]
fn should_render_json_report_for_scripts() {
    let sandbox_state = sandbox_instance_paths();
    write_config_file(&sandbox_state.instance_paths, MINIMAL_VALID_CONFIG_JSON);
    let mut rendered_output = Vec::new();
    let mut ignored_errors = Vec::new();
    run_validate_config(
        &ValidateConfigArguments {
            runtime_instance: AstronomicalRuntimeInstance::Development,
            render_json: true,
        },
        &mut ValidateConfigDependencies {
            instance_paths: sandbox_state.instance_paths.clone(),
            stdout: &mut rendered_output,
            stderr: &mut ignored_errors,
        },
    )
    .expect("valid config renders");
    let rendered_text = String::from_utf8(rendered_output).expect("utf-8 report");
    let report_document: serde_json::Value =
        serde_json::from_str(rendered_text.trim_end()).expect("json report parses");
    assert_eq!(report_document["runtime_instance"], "development");
    assert_eq!(
        report_document["model_directories"]
            .as_array()
            .expect("directories list")
            .len(),
        1
    );
    assert_eq!(
        report_document["configured_model_ids"]
            .as_array()
            .expect("model ids list")
            .len(),
        0
    );
    for report_key in [
        "configuration_file_path",
        "generation",
        "daemon_bind_address",
        "persistent_prompt_cache_enabled",
        "maximum_mlx_memory_bytes",
        "performance_attribution_enabled",
    ] {
        assert!(
            report_document.get(report_key).is_some(),
            "report is missing {report_key}"
        );
    }
}
