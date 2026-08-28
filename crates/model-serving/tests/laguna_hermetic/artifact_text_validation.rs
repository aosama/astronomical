use std::fs::{self, File};
use std::os::unix::fs::FileExt;
use std::os::unix::fs::symlink;

use astronomical_config::LagunaRootChatTemplateSelectionError;
use astronomical_model_serving::{
    ArtifactValidationError, LagunaArtifactValidationError, LagunaArtifactValidator,
    LagunaGenerationProcessor,
};
use serde_json::{Map, Value, json};

use super::artifact_support::{
    SYNTHETIC_BOS_TOKEN_ID, SYNTHETIC_EOS_TOKEN_ID, SYNTHETIC_PAD_TOKEN_ID, SyntheticLagunaArtifact,
};
use super::text_support::{
    LARGE_MODEL_CONTEXT_TOKEN_COUNT, POOLSIDE_TEMPLATE, ROMEO_AND_JULIET_SOURCE,
    SYNTHETIC_LAGUNA_MODEL_ID, romeo_and_juliet_command,
};

const REQUIRED_TEXT_SIDECAR_FILE_NAMES: [&str; 3] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
];
const INCLUDED_TEMPLATE_FILE_NAME: &str = "chat_template.jinja";

#[test]
fn should_validate_one_complete_weight_and_text_artifact_contract() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the complete synthetic artifact should validate");
    let text_artifact = validated_artifact.text_artifact();

    // The validator must return generation-ready semantics, not only weight metadata.
    assert_eq!(text_artifact.model_vocabulary_size(), 8);
    assert_eq!(text_artifact.maximum_context_tokens(), 32);
    assert_eq!(text_artifact.bos_token_id(), SYNTHETIC_BOS_TOKEN_ID);
    assert_eq!(text_artifact.pad_token_id(), SYNTHETIC_PAD_TOKEN_ID);
    assert_eq!(text_artifact.end_token_ids(), &[SYNTHETIC_EOS_TOKEN_ID]);
    assert_eq!(text_artifact.reasoning_parser_id(), "poolside_v1");
    assert_eq!(text_artifact.tool_call_parser_id(), "poolside_v1");
    assert!(!text_artifact.default_thinking_enabled());
    assert_eq!(
        text_artifact.generation_default_thinking_enabled(),
        Some(true)
    );
    assert!(text_artifact.sampler_config().uses_sampling());
}

#[test]
fn should_validate_and_prepare_romeo_and_juliet_from_a_standalone_root_template() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::dense("");
    fixture.config["max_position_embeddings"] = json!(LARGE_MODEL_CONTEXT_TOKEN_COUNT);
    fixture.write(model_directory.path());
    select_standalone_root_template(model_directory.path(), None);

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the standalone-root artifact should validate");
    let processor = LagunaGenerationProcessor::new(
        SYNTHETIC_LAGUNA_MODEL_ID,
        validated_artifact.text_artifact().clone(),
    )
    .expect("the standalone-root text descriptor should construct the processor");
    let prepared_generation = processor
        .prepare_chat(&romeo_and_juliet_command(9_899, None))
        .expect("the standalone-root Romeo and Juliet request should prepare");
    assert!(
        prepared_generation
            .rendered_prompt()
            .contains(ROMEO_AND_JULIET_SOURCE)
    );

    let retained_files = validated_artifact
        .into_retained_files()
        .expect("the standalone root descriptor should transfer");
    fs::remove_dir_all(model_directory.path()).expect("the fixture paths should be removable");
    let standalone_template_file = retained_files
        .standalone_chat_template_file()
        .expect("the selected standalone root descriptor should remain retained");
    assert_eq!(
        read_retained_file(standalone_template_file),
        POOLSIDE_TEMPLATE.as_bytes()
    );
    assert!(
        retained_files
            .included_template_files()
            .get(INCLUDED_TEMPLATE_FILE_NAME)
            .is_none()
    );
}

#[test]
fn should_accept_a_null_embedded_field_with_a_standalone_root_template() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    select_standalone_root_template(model_directory.path(), Some(Value::Null));

    LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("a null embedded field should defer to the standalone root");
}

#[test]
fn should_reject_conflicting_embedded_and_unselected_standalone_root_templates() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    fs::write(
        model_directory.path().join(INCLUDED_TEMPLATE_FILE_NAME),
        POOLSIDE_TEMPLATE,
    )
    .expect("the conflicting standalone root should be written");

    let validation_error = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect_err("two unconnected root authorities must be rejected");

    assert!(matches!(
        validation_error,
        LagunaArtifactValidationError::TemplateSource(
            LagunaRootChatTemplateSelectionError::ConflictingRootChatTemplates
        )
    ));
}

#[test]
fn should_reject_a_symlinked_standalone_root_template() {
    let fixture_home = tempfile::tempdir().expect("the test should create a fixture home");
    let model_directory = fixture_home.path().join("model");
    fs::create_dir(&model_directory).expect("the model directory should be created");
    SyntheticLagunaArtifact::dense("").write(&model_directory);
    select_standalone_root_template(&model_directory, None);
    fs::remove_file(model_directory.join(INCLUDED_TEMPLATE_FILE_NAME))
        .expect("the regular standalone root should be removed");
    let external_template_path = fixture_home.path().join("external-template.jinja");
    fs::write(&external_template_path, POOLSIDE_TEMPLATE)
        .expect("the external template should be written");
    symlink(
        &external_template_path,
        model_directory.join(INCLUDED_TEMPLATE_FILE_NAME),
    )
    .expect("the standalone template symlink should be created");

    let validation_error = LagunaArtifactValidator::new()
        .validate(&model_directory)
        .expect_err("ordinary standalone template symlinks must be rejected");

    assert!(matches!(
        validation_error,
        LagunaArtifactValidationError::Artifact(
            ArtifactValidationError::RequiredFileIsSymlink { file_name }
        ) if file_name == INCLUDED_TEMPLATE_FILE_NAME
    ));
}

#[test]
fn should_reject_a_standalone_root_that_includes_itself() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    select_standalone_root_template(model_directory.path(), None);
    fs::write(
        model_directory.path().join(INCLUDED_TEMPLATE_FILE_NAME),
        "{% include 'chat_template.jinja' %}",
    )
    .expect("the self-including standalone root should be written");

    let validation_error = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect_err("the standalone root must not reopen itself as an include");

    assert!(matches!(
        validation_error,
        LagunaArtifactValidationError::TextArtifact(
            astronomical_model_serving::LagunaTextArtifactError::TemplateIncludeCycle {
                include_name
            }
        ) if include_name == INCLUDED_TEMPLATE_FILE_NAME
    ));
}

#[test]
fn should_report_each_missing_required_text_sidecar_with_a_typed_cause() {
    for missing_sidecar_file_name in REQUIRED_TEXT_SIDECAR_FILE_NAMES {
        let model_directory =
            tempfile::tempdir().expect("the test should create a model directory");
        SyntheticLagunaArtifact::dense("").write(model_directory.path());
        fs::remove_file(model_directory.path().join(missing_sidecar_file_name))
            .expect("the selected required sidecar should be removed");

        let validation_error = LagunaArtifactValidator::new()
            .validate(model_directory.path())
            .expect_err("a complete Laguna artifact requires every text sidecar");

        assert!(matches!(
            validation_error,
            LagunaArtifactValidationError::Artifact(
                ArtifactValidationError::InspectRequiredFile { file_name, source }
            ) if file_name == missing_sidecar_file_name
                && source.kind() == std::io::ErrorKind::NotFound
        ));
    }
}

#[test]
fn should_resolve_an_included_template_from_its_validated_retained_descriptor() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::dense("").write(model_directory.path());
    select_included_template(model_directory.path());

    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the retained one-level template include should validate");
    let retained_files = validated_artifact
        .into_retained_files()
        .expect("every validated artifact descriptor should transfer");
    fs::remove_dir_all(model_directory.path()).expect("the fixture paths should be removable");

    assert!(!retained_files.text_artifact().default_thinking_enabled());
    let retained_template_file = retained_files
        .included_template_files()
        .get(INCLUDED_TEMPLATE_FILE_NAME)
        .expect("the selected included template descriptor should transfer");
    assert_eq!(
        read_retained_file(retained_template_file),
        POOLSIDE_TEMPLATE.as_bytes()
    );
}

#[test]
fn should_keep_text_descriptors_and_sidecars_usable_after_artifact_paths_disappear() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticLagunaArtifact::direct_affine_dense("", 2, 32, &[]).write(model_directory.path());
    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the affine artifact and its 256-token sidecars should validate");

    let retained_files = validated_artifact
        .into_retained_files()
        .expect("every validated text and weight descriptor should transfer");
    fs::remove_dir_all(model_directory.path()).expect("the fixture paths should be removable");

    assert_eq!(retained_files.text_artifact().model_vocabulary_size(), 256);
    // Processor construction after deletion proves the normalized descriptor owns tokenizer bytes.
    LagunaGenerationProcessor::new(
        SYNTHETIC_LAGUNA_MODEL_ID,
        retained_files.text_artifact().clone(),
    )
    .expect("the retained text descriptor should remain generation-ready");
    assert!(!read_retained_file(retained_files.tokenizer_file()).is_empty());
    assert!(!read_retained_file(retained_files.tokenizer_config_file()).is_empty());
    assert!(!read_retained_file(retained_files.generation_config_file()).is_empty());
    assert!(retained_files.standalone_chat_template_file().is_none());
}

#[test]
fn should_prepare_romeo_and_juliet_from_the_validator_owned_text_descriptor() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut fixture = SyntheticLagunaArtifact::dense("");
    fixture.config["max_position_embeddings"] = json!(LARGE_MODEL_CONTEXT_TOKEN_COUNT);
    fixture.write(model_directory.path());
    let validated_artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the complete synthetic text artifact should validate");
    let processor = LagunaGenerationProcessor::new(
        SYNTHETIC_LAGUNA_MODEL_ID,
        validated_artifact.text_artifact().clone(),
    )
    .expect("the validator-owned text descriptor should construct the processor");

    let prepared_generation = processor
        .prepare_chat(&romeo_and_juliet_command(9_898, None))
        .expect("the complete Romeo and Juliet request should prepare");

    assert!(!prepared_generation.prompt_token_ids().is_empty());
    assert!(
        prepared_generation
            .rendered_prompt()
            .contains(ROMEO_AND_JULIET_SOURCE)
    );
    assert!(prepared_generation.is_end_token(SYNTHETIC_EOS_TOKEN_ID));
}

/// Generates an exact-size WordLevel tokenizer so every mutable weight fixture remains valid.
pub(super) fn synthetic_text_sidecars(config: &Value) -> [(&'static str, Value); 3] {
    let vocabulary_size = u32::try_from(
        config["vocab_size"]
            .as_u64()
            .expect("the synthetic vocabulary size must be unsigned"),
    )
    .expect("the synthetic vocabulary size must fit u32");
    let maximum_context_tokens = config["max_position_embeddings"]
        .as_u64()
        .expect("the synthetic model context must be unsigned");
    let control_tokens = [
        (SYNTHETIC_BOS_TOKEN_ID, "<synthetic_bos>"),
        (SYNTHETIC_PAD_TOKEN_ID, "<synthetic_pad>"),
        (SYNTHETIC_EOS_TOKEN_ID, "<synthetic_eos>"),
    ];
    let token_descriptor = |token_id: u32, token_content: &str| {
        json!({"id": token_id, "content": token_content, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true})
    };
    let mut vocabulary = Map::new();
    for token_id in 0..vocabulary_size {
        vocabulary.insert(format!("token_{token_id}"), json!(token_id));
    }
    let mut added_tokens_decoder = Map::new();
    let added_tokens = control_tokens.map(|(token_id, token_content)| {
        vocabulary.remove(&format!("token_{token_id}"));
        vocabulary.insert(token_content.to_owned(), json!(token_id));
        let descriptor = token_descriptor(token_id, token_content);
        added_tokens_decoder.insert(token_id.to_string(), descriptor.clone());
        descriptor
    });
    let tokenizer = json!({"version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens, "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "token_0"}});
    let tokenizer_config = json!({"added_tokens_decoder": added_tokens_decoder,
        "bos_token": "<synthetic_bos>", "pad_token": "<synthetic_pad>",
        "eos_token": "<synthetic_eos>", "model_max_length": maximum_context_tokens,
        "chat_template": POOLSIDE_TEMPLATE});
    let generation_config = json!({"do_sample": true, "bos_token_id": SYNTHETIC_BOS_TOKEN_ID,
        "pad_token_id": SYNTHETIC_PAD_TOKEN_ID, "eos_token_id": [SYNTHETIC_EOS_TOKEN_ID],
        "temperature": 1.0, "top_p": 1.0, "reasoning_parser": "poolside_v1",
        "tool_call_parser": "poolside_v1",
        "default_chat_template_kwargs": {"enable_thinking": true}});
    [
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", tokenizer_config),
        ("generation_config.json", generation_config),
    ]
}

fn select_included_template(model_directory: &std::path::Path) {
    let tokenizer_config_path = model_directory.join("tokenizer_config.json");
    let mut tokenizer_config: Value = serde_json::from_slice(
        &fs::read(&tokenizer_config_path).expect("the tokenizer config should be readable"),
    )
    .expect("the tokenizer config should be valid JSON");
    tokenizer_config["chat_template"] = json!("{% include 'chat_template.jinja' %}");
    fs::write(
        tokenizer_config_path,
        serde_json::to_vec(&tokenizer_config)
            .expect("the included-template tokenizer config should serialize"),
    )
    .expect("the included-template tokenizer config should be written");
    fs::write(
        model_directory.join(INCLUDED_TEMPLATE_FILE_NAME),
        POOLSIDE_TEMPLATE,
    )
    .expect("the included template should be written");
}

fn select_standalone_root_template(
    model_directory: &std::path::Path,
    embedded_template: Option<Value>,
) {
    let tokenizer_config_path = model_directory.join("tokenizer_config.json");
    let mut tokenizer_config: Value = serde_json::from_slice(
        &fs::read(&tokenizer_config_path).expect("the tokenizer config should be readable"),
    )
    .expect("the tokenizer config should be valid JSON");
    let tokenizer_config_fields = tokenizer_config
        .as_object_mut()
        .expect("the synthetic tokenizer config should be an object");
    match embedded_template {
        Some(embedded_template) => {
            tokenizer_config_fields.insert("chat_template".to_owned(), embedded_template);
        }
        None => {
            tokenizer_config_fields.remove("chat_template");
        }
    }
    fs::write(
        tokenizer_config_path,
        serde_json::to_vec(&tokenizer_config)
            .expect("the standalone-root tokenizer config should serialize"),
    )
    .expect("the standalone-root tokenizer config should be written");
    fs::write(
        model_directory.join(INCLUDED_TEMPLATE_FILE_NAME),
        POOLSIDE_TEMPLATE,
    )
    .expect("the standalone root template should be written");
}

fn read_retained_file(retained_file: &File) -> Vec<u8> {
    let file_size = usize::try_from(
        retained_file
            .metadata()
            .expect("the retained file metadata should remain readable")
            .len(),
    )
    .expect("the retained synthetic file size should fit usize");
    let mut file_bytes = vec![0_u8; file_size];
    retained_file
        .read_exact_at(&mut file_bytes, 0)
        .expect("the retained descriptor bytes should remain readable");
    file_bytes
}
