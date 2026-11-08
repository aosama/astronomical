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
    assert!(rendered_text.contains("Neural Engine programs: "));
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
    assert!(
        report_document["neural_engine_program_directory"]
            .as_str()
            .expect("the folder is reported")
            .ends_with("neural-engine-programs")
    );
    assert_eq!(
        report_document["model_directories"]
            .as_array()
            .expect("directories list")
            .len(),
        1
    );
    assert_eq!(
        report_document["model_settings"],
        serde_json::json!({"checked": true, "problem": null, "entries": []})
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

/// One installed model as a model download publishes it under `models_root`: the model files
/// plus the provenance file that records its full Hugging Face ID.
fn install_library_model(models_root: &std::path::Path, provider_model_id: &str) {
    const MODEL_SHARD_BYTES: &[u8] = b"fictional-shard";
    let model_directory = models_root.join(provider_model_id);
    std::fs::create_dir_all(&model_directory).expect("model directory");
    let model_files = [
        (
            "config.json",
            r#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#
                .to_owned(),
        ),
        (
            "model.safetensors.index.json",
            format!(
                r#"{{"metadata":{{"total_size":{}}},"weight_map":{{"model.embed_tokens.weight":"model-00001.safetensors"}}}}"#,
                MODEL_SHARD_BYTES.len()
            ),
        ),
        (
            "tokenizer.json",
            r#"{"version":1,"model":{"type":"BPE"}}"#.to_owned(),
        ),
        (
            ".astronomical-library-provenance.json",
            format!(
                r#"{{"provider_model_id":"{provider_model_id}","revision":"0123456789abcdef0123456789abcdef01234567","schema_version":1}}"#
            ),
        ),
    ];
    for (file_name, file_contents) in model_files {
        std::fs::write(model_directory.join(file_name), file_contents).expect("model file");
    }
    std::fs::write(
        model_directory.join("model-00001.safetensors"),
        MODEL_SHARD_BYTES,
    )
    .expect("model shard");
}

fn config_with_models(
    model_directories: &[&std::path::Path],
    configured_model_ids: &[&str],
) -> String {
    let mut models_document = serde_json::Map::new();
    for configured_model_id in configured_model_ids {
        models_document.insert(
            (*configured_model_id).to_owned(),
            serde_json::json!({"generation_defaults": {"temperature": 0.5}}),
        );
    }
    serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {"model_directories": model_directories},
        "models": models_document,
    })
    .to_string()
}

struct ValidateConfigOutput {
    stdout: String,
    stderr: String,
}

fn validate_output(
    instance_paths: &AstronomicalInstancePaths,
    render_json: bool,
) -> ValidateConfigOutput {
    let mut rendered_output = Vec::new();
    let mut rendered_warnings = Vec::new();
    run_validate_config(
        &ValidateConfigArguments {
            runtime_instance: AstronomicalRuntimeInstance::Development,
            render_json,
        },
        &mut ValidateConfigDependencies {
            instance_paths: instance_paths.clone(),
            stdout: &mut rendered_output,
            stderr: &mut rendered_warnings,
        },
    )
    .expect("a valid config validates even when models are missing");
    ValidateConfigOutput {
        stdout: String::from_utf8(rendered_output).expect("utf-8 report"),
        stderr: String::from_utf8(rendered_warnings).expect("utf-8 warnings"),
    }
}

/// #1925: the report says what each `models` entry does, by the same rule the daemon applies,
/// without the daemon running.
#[test]
fn should_say_which_installed_model_each_entry_is_used_for() {
    let sandbox_state = sandbox_instance_paths();
    install_library_model(
        &sandbox_state.instance_paths.models_directory(),
        "astronomical-test/example-qwen",
    );
    write_config_file(
        &sandbox_state.instance_paths,
        &config_with_models(
            &[],
            &[
                "astronomical-test/example-qwen",
                "other-organisation/example-qwen",
                "someorg/not-installed",
            ],
        ),
    );

    let validate_report = validate_output(&sandbox_state.instance_paths, false);

    let rendered_text = validate_report.stdout;
    assert!(
        rendered_text.contains("Model settings: 3"),
        "{rendered_text}"
    );
    assert!(
        rendered_text.contains("astronomical-test/example-qwen: used for example-qwen"),
        "{rendered_text}"
    );
    assert!(
        rendered_text.contains(
            "other-organisation/example-qwen: not used: your installed example-qwen is astronomical-test/example-qwen. Use that name instead."
        ),
        "{rendered_text}"
    );
    assert!(
        rendered_text
            .contains("someorg/not-installed: waiting: no installed model has this name yet."),
        "{rendered_text}"
    );
    assert!(validate_report.stderr.is_empty());
}

#[test]
fn should_find_models_in_configured_folders_and_by_short_name() {
    let sandbox_state = sandbox_instance_paths();
    let configured_root = sandbox_state
        .instance_paths
        .config_file_path()
        .with_file_name("extra-models");
    install_library_model(&configured_root, "astronomical-test/example-qwen");
    write_config_file(
        &sandbox_state.instance_paths,
        &config_with_models(&[configured_root.as_path()], &["example-qwen"]),
    );

    let rendered_text = validate_output(&sandbox_state.instance_paths, false).stdout;

    assert!(
        rendered_text.contains("example-qwen: used for example-qwen"),
        "{rendered_text}"
    );
}

#[test]
fn should_describe_each_entry_for_scripts_in_json() {
    let sandbox_state = sandbox_instance_paths();
    install_library_model(
        &sandbox_state.instance_paths.models_directory(),
        "astronomical-test/example-qwen",
    );
    write_config_file(
        &sandbox_state.instance_paths,
        &config_with_models(
            &[],
            &["astronomical-test/example-qwen", "someorg/not-installed"],
        ),
    );

    let report_document: serde_json::Value =
        serde_json::from_str(&validate_output(&sandbox_state.instance_paths, true).stdout)
            .expect("json report");

    assert_eq!(
        report_document["model_settings"],
        serde_json::json!({
            "checked": true,
            "problem": null,
            "entries": [
                {"configured_model_id": "astronomical-test/example-qwen", "outcome": "applies", "model_id": "example-qwen"},
                {"configured_model_id": "someorg/not-installed", "outcome": "matches_no_installed_model"},
            ],
        })
    );
}

/// Two installed models with the same short name stop the daemon from loading; validation still
/// reads the document, but says plainly why the entries could not be checked.
#[test]
fn should_explain_in_plain_words_when_the_installed_models_cannot_be_listed() {
    let sandbox_state = sandbox_instance_paths();
    let models_directory = sandbox_state.instance_paths.models_directory();
    install_library_model(&models_directory, "astronomical-test/example-qwen");
    install_library_model(&models_directory, "other-organisation/example-qwen");
    write_config_file(
        &sandbox_state.instance_paths,
        &config_with_models(&[], &["example-qwen"]),
    );

    let text_report = validate_output(&sandbox_state.instance_paths, false);
    let json_report = validate_output(&sandbox_state.instance_paths, true);

    assert!(
        text_report
            .stdout
            .contains("two installed models are both called example-qwen"),
        "{}",
        text_report.stdout
    );
    assert!(
        text_report
            .stderr
            .starts_with("warning: the model settings were not checked:"),
        "{}",
        text_report.stderr
    );
    let report_document: serde_json::Value =
        serde_json::from_str(&json_report.stdout).expect("json report");
    assert_eq!(report_document["model_settings"]["checked"], false);
    assert!(
        report_document["model_settings"]["problem"]
            .as_str()
            .is_some_and(|problem| problem.contains("both called example-qwen"))
    );
    assert_eq!(
        report_document["model_settings"]["entries"][0]["outcome"],
        "not_checked"
    );
}

#[test]
fn should_say_none_and_skip_the_scan_when_no_model_settings_exist() {
    let sandbox_state = sandbox_instance_paths();
    write_config_file(&sandbox_state.instance_paths, MINIMAL_VALID_CONFIG_JSON);

    let rendered_text = validate_output(&sandbox_state.instance_paths, false).stdout;

    assert!(
        rendered_text.contains("Model settings: none"),
        "{rendered_text}"
    );
}
