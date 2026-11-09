use std::ffi::OsString;
use std::path::Path;
use std::time::{Duration, Instant};

use astronomical_expert_atlas::{ConversionProgress, ConversionStage};

use super::{
    ConvertAtlasArguments, ConvertAtlasDependencies, ConvertAtlasError, ResolvedSource,
    parse_convert_atlas_arguments, progress_line, resolve_source, run_convert_atlas,
};
use crate::UsageError;

fn os_arguments(arguments: &[&str]) -> Vec<OsString> {
    arguments.iter().map(OsString::from).collect()
}

/// A one-shard model with one expert layer of two experts and one resident
/// tensor, at `library/organization/name`.
fn write_tiny_model(model_directory: &Path) {
    std::fs::create_dir_all(model_directory).expect("model directory");
    let tensors = [
        (
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            vec![2_usize, 8],
            vec![3_u8; 16],
        ),
        ("model.embed_tokens.weight", vec![4], vec![9_u8; 4]),
    ];
    let mut header_document = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut payload_offset = 0;
    for (tensor_name, shape, tensor_bytes) in &tensors {
        header_document.insert((*tensor_name).to_owned(), serde_json::json!({"dtype": "U8", "shape": shape, "data_offsets": [payload_offset, payload_offset + tensor_bytes.len()]}));
        weight_map.insert(
            (*tensor_name).to_owned(),
            serde_json::json!("model-00001.safetensors"),
        );
        payload_offset += tensor_bytes.len();
    }
    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header_document)).expect("header");
    let mut shard_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    shard_bytes.extend(header_bytes);
    for (_, _, tensor_bytes) in &tensors {
        shard_bytes.extend(tensor_bytes);
    }
    std::fs::write(model_directory.join("model-00001.safetensors"), shard_bytes).expect("shard");
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).expect("index"),
    )
    .expect("index writes");
    std::fs::write(
        model_directory.join("config.json"),
        br#"{"num_experts": 2}"#,
    )
    .expect("config");
}

fn run_in_library(
    models_directory: &Path,
    convert_arguments: &ConvertAtlasArguments,
) -> (Result<(), ConvertAtlasError>, String, String) {
    let mut stdout_bytes = Vec::new();
    let mut stderr_bytes = Vec::new();
    let conversion_outcome = run_convert_atlas(
        convert_arguments,
        ConvertAtlasDependencies {
            models_directory: models_directory.to_path_buf(),
            stdout: &mut stdout_bytes,
            stderr: &mut stderr_bytes,
            progress_interval: Duration::ZERO,
        },
    );
    (
        conversion_outcome,
        String::from_utf8_lossy(&stdout_bytes).into_owned(),
        String::from_utf8_lossy(&stderr_bytes).into_owned(),
    )
}

#[test]
fn should_parse_a_model_and_an_optional_output_model_id() {
    assert_eq!(
        parse_convert_atlas_arguments(&os_arguments(&["org/model"])).expect("parses"),
        ConvertAtlasArguments {
            source_model: "org/model".to_owned(),
            output_model_id: None
        }
    );
    assert_eq!(
        parse_convert_atlas_arguments(&os_arguments(&[
            "org/model",
            "--output-model-id",
            "org/other"
        ]))
        .expect("parses"),
        ConvertAtlasArguments {
            source_model: "org/model".to_owned(),
            output_model_id: Some("org/other".to_owned())
        }
    );
}

#[test]
fn should_refuse_a_missing_model_an_unknown_flag_and_a_second_model() {
    assert!(matches!(
        parse_convert_atlas_arguments(&[]),
        Err(UsageError::ConvertAtlasModelRequired)
    ));
    assert!(matches!(
        parse_convert_atlas_arguments(&os_arguments(&["org/model", "--force"])),
        Err(UsageError::UnknownArgument(_))
    ));
    assert!(matches!(
        parse_convert_atlas_arguments(&os_arguments(&["org/a", "org/b"])),
        Err(UsageError::UnknownArgument(_))
    ));
    assert!(matches!(
        parse_convert_atlas_arguments(&os_arguments(&["org/model", "--output-model-id"])),
        Err(UsageError::MissingValue(_))
    ));
}

#[test]
fn should_resolve_a_library_id_and_a_folder_to_the_same_model_id() {
    let library = tempfile::tempdir().expect("scratch library");
    let model_directory = library.path().join("organization/model-name");
    write_tiny_model(&model_directory);
    let by_id = resolve_source("organization/model-name", library.path()).expect("resolves by id");
    assert_eq!(
        by_id,
        ResolvedSource {
            source_directory: model_directory.clone(),
            source_model_id: "organization/model-name".to_owned()
        }
    );
    let by_folder = resolve_source(
        model_directory.to_str().expect("utf-8 path"),
        Path::new("/nonexistent"),
    )
    .expect("resolves by folder");
    assert_eq!(by_folder.source_model_id, "organization/model-name");
}

#[test]
fn should_refuse_an_absent_model_and_an_id_that_escapes_the_library() {
    let library = tempfile::tempdir().expect("scratch library");
    assert!(matches!(
        resolve_source("organization/absent", library.path()),
        Err(ConvertAtlasError::SourceNotFound { .. })
    ));
    assert!(matches!(
        resolve_source("../outside", library.path()),
        Err(ConvertAtlasError::InvalidModelId(_))
    ));
}

#[test]
fn should_convert_into_the_library_under_the_default_atlas_name() {
    let library = tempfile::tempdir().expect("scratch library");
    write_tiny_model(&library.path().join("organization/model-name"));
    let (conversion_outcome, stdout_text, stderr_text) = run_in_library(
        library.path(),
        &ConvertAtlasArguments {
            source_model: "organization/model-name".to_owned(),
            output_model_id: None,
        },
    );
    conversion_outcome.expect("the conversion succeeds");
    assert!(
        library
            .path()
            .join("organization/model-name-astro-atlas-streaming/manifest.json")
            .is_file()
    );
    assert!(
        stdout_text.contains("into organization/model-name-astro-atlas-streaming"),
        "{stdout_text}"
    );
    assert!(
        stderr_text.contains("[convert-atlas] layer 1/1 (100%)"),
        "{stderr_text}"
    );
}

#[test]
fn should_convert_under_a_chosen_output_model_id_and_refuse_to_overwrite_it() {
    let library = tempfile::tempdir().expect("scratch library");
    write_tiny_model(&library.path().join("organization/model-name"));
    let convert_arguments = ConvertAtlasArguments {
        source_model: "organization/model-name".to_owned(),
        output_model_id: Some("mine/fast-model".to_owned()),
    };
    run_in_library(library.path(), &convert_arguments)
        .0
        .expect("the first conversion succeeds");
    assert!(
        library
            .path()
            .join("mine/fast-model/manifest.json")
            .is_file()
    );
    assert!(matches!(
        run_in_library(library.path(), &convert_arguments).0,
        Err(ConvertAtlasError::Conversion(_))
    ));
}

#[test]
fn should_describe_resident_and_layer_progress_with_a_percentage() {
    let mut conversion_progress = ConversionProgress {
        stage: ConversionStage::ResidentBundle,
        written_byte_count: 1_000_000_000,
        total_byte_count: 4_000_000_000,
        completed_layer_count: 0,
        total_layer_count: 40,
        started_at: Instant::now(),
    };
    assert!(
        progress_line(&conversion_progress)
            .starts_with("[convert-atlas] resident weights (25%) 1.0 of 4.0 GB, ETA")
    );
    conversion_progress.stage = ConversionStage::ExpertLayers;
    conversion_progress.completed_layer_count = 12;
    assert!(progress_line(&conversion_progress).starts_with("[convert-atlas] layer 12/40 (25%)"));
}
