use std::fs;
use std::path::Path;

use astronomical_config::ModelFamily;

use super::discover_configured_models;

const IMMUTABLE_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn should_advertise_one_complete_executable_laguna_artifact() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Laguna-Fixture");
    write_executable_laguna_artifact(&model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .first()
        .expect("the complete Laguna artifact should be advertised");

    assert_eq!(directory_scans[0].discovered_models.len(), 1);
    assert_eq!(discovered_model.model_family, ModelFamily::Laguna);
    assert_eq!(discovered_model.revision, IMMUTABLE_REVISION);
    assert_eq!(discovered_model.context_window, 65_536);
    assert_eq!(discovered_model.max_input_tokens, 45_056);
    assert_eq!(discovered_model.max_output_tokens, 20_480);
    assert!(!discovered_model.has_vision);
    assert!(discovered_model.supports_reasoning);
    assert!(discovered_model.supports_tool_calls);
    assert_eq!(discovered_model.model_size_bytes, 96);
}

#[test]
fn should_require_every_laguna_discovery_boundary_before_advertising() {
    for missing_relative_path in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001.safetensors",
        ".cache/huggingface/download/config.json.metadata",
    ] {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        let model_directory = temporary_directory.path().join("Laguna-Incomplete");
        write_executable_laguna_artifact(&model_directory);
        fs::remove_file(model_directory.join(missing_relative_path))
            .expect("the selected required file should be removed");

        assert!(
            discover_configured_models(&temporary_directory)[0]
                .discovered_models
                .is_empty(),
            "Laguna discovery must reject an artifact missing {missing_relative_path}"
        );
    }
}

#[test]
fn should_reject_unsupported_text_contract_and_missing_template_include() {
    let unsupported_parser_home =
        tempfile::tempdir().expect("temporary directory should be created");
    let unsupported_parser_directory = unsupported_parser_home.path().join("Unsupported-Parser");
    write_executable_laguna_artifact(&unsupported_parser_directory);
    fs::write(
        unsupported_parser_directory.join("generation_config.json"),
        r#"{"reasoning_parser":"unknown","tool_call_parser":"poolside_v1"}"#,
    )
    .expect("unsupported generation config should be written");
    assert!(
        discover_configured_models(&unsupported_parser_home)[0]
            .discovered_models
            .is_empty()
    );

    let missing_include_home = tempfile::tempdir().expect("temporary directory should be created");
    let missing_include_directory = missing_include_home.path().join("Missing-Include");
    write_executable_laguna_artifact(&missing_include_directory);
    fs::write(
        missing_include_directory.join("tokenizer_config.json"),
        r#"{"chat_template":"{% include 'missing.jinja' %}"}"#,
    )
    .expect("include-bearing tokenizer config should be written");
    assert!(
        discover_configured_models(&missing_include_home)[0]
            .discovered_models
            .is_empty()
    );
}

#[test]
fn should_reject_unsafe_incomplete_or_zero_payload_laguna_indexes() {
    for index_document in [
        r#"{"metadata":{"total_size":96},"weight_map":{"model.embed_tokens.weight":"../outside.safetensors"}}"#,
        r#"{"metadata":{"total_size":96},"weight_map":{"model.embed_tokens.weight":"missing.safetensors"}}"#,
        r#"{"metadata":{"total_size":0},"weight_map":{}}"#,
        r#"{"metadata":{"total_size":97},"weight_map":{"model.embed_tokens.weight":"model-00001.safetensors"}}"#,
    ] {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        let model_directory = temporary_directory.path().join("Laguna-Unsafe-Index");
        write_executable_laguna_artifact(&model_directory);
        fs::write(
            model_directory.join("model.safetensors.index.json"),
            index_document,
        )
        .expect("the selected unsafe index should be written");

        assert!(
            discover_configured_models(&temporary_directory)[0]
                .discovered_models
                .is_empty(),
            "Laguna discovery must reject index {index_document}"
        );
    }
}

#[test]
fn should_reject_nonimmutable_revision_and_zero_context() {
    let mutable_revision_home = tempfile::tempdir().expect("temporary directory should be created");
    let mutable_revision_directory = mutable_revision_home.path().join("Mutable-Revision");
    write_executable_laguna_artifact(&mutable_revision_directory);
    fs::write(
        mutable_revision_directory.join(".cache/huggingface/download/config.json.metadata"),
        "main\n",
    )
    .expect("mutable revision metadata should be written");
    assert!(
        discover_configured_models(&mutable_revision_home)[0]
            .discovered_models
            .is_empty()
    );

    let zero_context_home = tempfile::tempdir().expect("temporary directory should be created");
    let zero_context_directory = zero_context_home.path().join("Zero-Context");
    write_executable_laguna_artifact(&zero_context_directory);
    fs::write(
        zero_context_directory.join("config.json"),
        r#"{"model_type":"laguna","text_config":{"max_position_embeddings":0}}"#,
    )
    .expect("zero-context model config should be written");
    assert!(
        discover_configured_models(&zero_context_home)[0]
            .discovered_models
            .is_empty()
    );
}

fn write_executable_laguna_artifact(model_directory: &Path) {
    fs::create_dir_all(model_directory.join(".cache/huggingface/download"))
        .expect("Laguna fixture directories should be created");
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"laguna","text_config":{"max_position_embeddings":65536}}"#,
    )
    .expect("Laguna model config should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE"}}"#,
    )
    .expect("Laguna tokenizer should be written");
    fs::write(
        model_directory.join("tokenizer_config.json"),
        r#"{"chat_template":"{{ messages }}"}"#,
    )
    .expect("Laguna tokenizer config should be written");
    fs::write(
        model_directory.join("generation_config.json"),
        r#"{"reasoning_parser":"poolside_v1","tool_call_parser":"poolside_v1"}"#,
    )
    .expect("Laguna generation config should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":96},"weight_map":{"model.embed_tokens.weight":"model-00001.safetensors"}}"#,
    )
    .expect("Laguna shard index should be written");
    fs::write(
        model_directory.join("model-00001.safetensors"),
        vec![0_u8; 96],
    )
    .expect("Laguna model shard should be written");
    fs::write(
        model_directory.join(".cache/huggingface/download/config.json.metadata"),
        format!("{IMMUTABLE_REVISION}\n"),
    )
    .expect("Laguna immutable revision should be written");
}
