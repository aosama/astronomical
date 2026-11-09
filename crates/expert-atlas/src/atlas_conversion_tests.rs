//! End-to-end conversion of a small synthetic checkpoint: two expert layers
//! of four experts, split across two shards, plus resident tensors.

use std::collections::BTreeMap;
use std::path::Path;

use super::{
    AtlasConversionError, ConversionProgress, ConversionProgressObserver, ConversionRequest,
    convert_to_atlas,
};
use crate::atlas_manifest::{AtlasManifest, RESIDENT_BUNDLE_FILE_NAME};

const EXPERT_COUNT: usize = 4;
const EXPERT_TENSOR_SUFFIXES: [&str; 3] = [
    "switch_mlp.gate_proj.weight",
    "switch_mlp.gate_proj.scales",
    "switch_mlp.down_proj.weight",
];

/// The bytes of one synthetic tensor: each byte names its tensor and position.
fn synthetic_tensor_bytes(tensor_seed: u8, byte_count: usize) -> Vec<u8> {
    (0..byte_count)
        .map(|byte_index| tensor_seed.wrapping_mul(31).wrapping_add(byte_index as u8))
        .collect()
}

/// `(name, dtype, shape, bytes)` for every tensor of the synthetic model.
fn synthetic_tensors() -> Vec<(String, String, Vec<usize>, Vec<u8>)> {
    let mut tensors = Vec::new();
    for layer_index in 0..2 {
        for (suffix_index, suffix) in EXPERT_TENSOR_SUFFIXES.iter().enumerate() {
            let per_expert_bytes = 1_000 + 24 * suffix_index;
            let seed = (layer_index * 10 + suffix_index) as u8;
            tensors.push((
                format!("language_model.model.layers.{layer_index}.mlp.{suffix}"),
                "U8".to_owned(),
                vec![EXPERT_COUNT, per_expert_bytes],
                synthetic_tensor_bytes(seed, EXPERT_COUNT * per_expert_bytes),
            ));
        }
        tensors.push((
            format!("language_model.model.layers.{layer_index}.self_attn.q_proj.weight"),
            "U8".to_owned(),
            vec![64],
            synthetic_tensor_bytes(100 + layer_index as u8, 64),
        ));
    }
    tensors.push((
        "language_model.model.embed_tokens.weight".to_owned(),
        "U8".to_owned(),
        vec![96],
        synthetic_tensor_bytes(200, 96),
    ));
    tensors
}

fn write_safetensors_shard(
    shard_path: &Path,
    shard_tensors: &[&(String, String, Vec<usize>, Vec<u8>)],
) {
    let mut header_document = serde_json::Map::new();
    header_document.insert(
        "__metadata__".to_owned(),
        serde_json::json!({"format": "mlx"}),
    );
    let mut payload_offset = 0;
    for (tensor_name, dtype, shape, tensor_bytes) in shard_tensors {
        header_document.insert(tensor_name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [payload_offset, payload_offset + tensor_bytes.len()]}));
        payload_offset += tensor_bytes.len();
    }
    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header_document)).expect("header encodes");
    let mut shard_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    shard_bytes.extend(header_bytes);
    for (_, _, _, tensor_bytes) in shard_tensors {
        shard_bytes.extend(tensor_bytes);
    }
    std::fs::write(shard_path, shard_bytes).expect("shard writes");
}

/// A model directory with two shards (tensors alternate between them), an
/// index, a config naming the expert count, a tokenizer and a hidden file.
fn write_synthetic_model(model_directory: &Path) {
    std::fs::create_dir_all(model_directory).expect("model directory");
    let tensors = synthetic_tensors();
    let mut weight_map = BTreeMap::new();
    for shard_number in 0..2 {
        let shard_file_name = format!("model-0000{}.safetensors", shard_number + 1);
        let shard_tensors: Vec<_> = tensors
            .iter()
            .enumerate()
            .filter(|(tensor_index, _)| tensor_index % 2 == shard_number)
            .map(|(_, tensor)| tensor)
            .collect();
        for (tensor_name, _, _, _) in &shard_tensors {
            weight_map.insert(tensor_name.clone(), shard_file_name.clone());
        }
        write_safetensors_shard(&model_directory.join(&shard_file_name), &shard_tensors);
    }
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).expect("index"),
    )
    .expect("index writes");
    std::fs::write(
        model_directory.join("config.json"),
        br#"{"text_config": {"num_experts": 4}}"#,
    )
    .expect("config writes");
    std::fs::write(model_directory.join("tokenizer.json"), b"{}").expect("tokenizer writes");
    std::fs::write(
        model_directory.join(".astronomical-library-provenance.json"),
        br#"{"provider_model_id": "example/model", "revision": "abc123"}"#,
    )
    .expect("provenance writes");
}

struct RecordingObserver {
    observed_progress: Vec<ConversionProgress>,
}

impl ConversionProgressObserver for RecordingObserver {
    fn observe(&mut self, conversion_progress: &ConversionProgress) {
        self.observed_progress.push(*conversion_progress);
    }
}

fn convert(
    source_directory: &Path,
    output_directory: &Path,
) -> Result<super::ConversionSummary, AtlasConversionError> {
    let mut recording_observer = RecordingObserver {
        observed_progress: Vec::new(),
    };
    let conversion_outcome = convert_to_atlas(
        &ConversionRequest {
            source_directory,
            source_model_id: "example/model",
            output_directory,
        },
        &mut recording_observer,
    );
    if conversion_outcome.is_ok() {
        let final_progress = recording_observer
            .observed_progress
            .last()
            .expect("progress is reported");
        assert_eq!(
            final_progress.written_byte_count,
            final_progress.total_byte_count
        );
        assert_eq!(final_progress.completed_layer_count, 2);
    }
    conversion_outcome
}

#[test]
fn should_place_every_expert_slice_in_its_record_byte_for_byte() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory
        .path()
        .join("example/model-astro-atlas-streaming");
    write_synthetic_model(&source_directory);
    convert(&source_directory, &output_directory).expect("the conversion succeeds");
    let atlas_manifest =
        AtlasManifest::read_from_directory(&output_directory).expect("the manifest validates");
    assert_eq!(atlas_manifest.source.revision.as_deref(), Some("abc123"));
    for (tensor_name, _, shape, tensor_bytes) in synthetic_tensors()
        .iter()
        .filter(|tensor| tensor.0.contains("switch_mlp"))
    {
        let layer_index = crate::source_checkpoint::routed_expert_layer_index(tensor_name)
            .expect("an expert tensor");
        let expert_layer = atlas_manifest
            .expert_layer(layer_index)
            .expect("the layer is declared");
        let slot = expert_layer
            .slots
            .iter()
            .find(|slot| &slot.tensor_name == tensor_name)
            .expect("the slot is declared");
        let layer_bytes = std::fs::read(output_directory.join(&expert_layer.file_name))
            .expect("the layer file reads");
        let bytes_per_expert = shape[1];
        for expert_id in 0..EXPERT_COUNT {
            let record_start = expert_layer.slot_file_offset(expert_id, slot) as usize;
            assert_eq!(
                &layer_bytes[record_start..record_start + bytes_per_expert],
                &tensor_bytes[expert_id * bytes_per_expert..(expert_id + 1) * bytes_per_expert],
                "{tensor_name} expert {expert_id}"
            );
        }
    }
}

#[test]
fn should_carry_every_resident_tensor_and_companion_file_but_no_shard() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory.path().join("example/converted");
    write_synthetic_model(&source_directory);
    let conversion_summary =
        convert(&source_directory, &output_directory).expect("the conversion succeeds");
    assert_eq!(conversion_summary.resident_tensor_count, 3);
    let bundle_bytes =
        std::fs::read(output_directory.join(RESIDENT_BUNDLE_FILE_NAME)).expect("the bundle reads");
    let header_byte_count =
        u64::from_le_bytes(bundle_bytes[..8].try_into().expect("length prefix")) as usize;
    let header_document: serde_json::Value =
        serde_json::from_slice(&bundle_bytes[8..8 + header_byte_count]).expect("the header parses");
    for (tensor_name, _, _, tensor_bytes) in synthetic_tensors()
        .iter()
        .filter(|tensor| !tensor.0.contains("switch_mlp"))
    {
        let data_offsets = &header_document[tensor_name]["data_offsets"];
        let payload_start =
            8 + header_byte_count + data_offsets[0].as_u64().expect("start") as usize;
        assert_eq!(
            &bundle_bytes[payload_start..payload_start + tensor_bytes.len()],
            tensor_bytes.as_slice(),
            "{tensor_name}"
        );
    }
    assert_eq!(header_document["__metadata__"]["format"], "mlx");
    assert!(output_directory.join("tokenizer.json").is_file());
    assert!(output_directory.join("config.json").is_file());
    assert!(!output_directory.join("model-00001.safetensors").exists());
    assert!(
        !output_directory
            .join("model.safetensors.index.json")
            .exists()
    );
    assert!(
        !output_directory
            .join(".astronomical-library-provenance.json")
            .exists()
    );
}

#[test]
fn should_refuse_an_existing_output_and_leave_it_untouched() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory.path().join("example/taken");
    write_synthetic_model(&source_directory);
    std::fs::create_dir_all(&output_directory).expect("an existing output");
    std::fs::write(output_directory.join("keep.txt"), b"mine").expect("an existing file");
    assert!(matches!(
        convert(&source_directory, &output_directory),
        Err(AtlasConversionError::OutputExists(_))
    ));
    assert_eq!(
        std::fs::read(output_directory.join("keep.txt")).expect("still there"),
        b"mine"
    );
}

#[test]
fn should_refuse_a_model_without_routed_experts_and_publish_nothing() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/dense");
    let output_directory = scratch_directory
        .path()
        .join("example/dense-astro-atlas-streaming");
    std::fs::create_dir_all(&source_directory).expect("model directory");
    let dense_tensor = (
        "model.embed_tokens.weight".to_owned(),
        "U8".to_owned(),
        vec![8],
        vec![7u8; 8],
    );
    write_safetensors_shard(
        &source_directory.join("model-00001.safetensors"),
        &[&dense_tensor],
    );
    std::fs::write(
        source_directory.join("model.safetensors.index.json"),
        br#"{"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}}"#,
    )
    .expect("index");
    std::fs::write(
        source_directory.join("config.json"),
        br#"{"num_experts": 4}"#,
    )
    .expect("config");
    assert!(matches!(
        convert(&source_directory, &output_directory),
        Err(AtlasConversionError::Source(_))
    ));
    assert!(!output_directory.exists());
}

#[test]
fn should_replace_a_staging_directory_left_by_an_interrupted_conversion() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory
        .path()
        .join("example/model-astro-atlas-streaming");
    write_synthetic_model(&source_directory);
    let stale_staging_directory = scratch_directory
        .path()
        .join("example/.model-astro-atlas-streaming.atlas-partial");
    std::fs::create_dir_all(stale_staging_directory.join("experts"))
        .expect("a stale staging directory");
    std::fs::write(
        stale_staging_directory.join("experts/layer-000.atlas"),
        b"half written",
    )
    .expect("a stale layer");
    convert(&source_directory, &output_directory).expect("the conversion succeeds");
    assert!(!stale_staging_directory.exists());
    AtlasManifest::read_from_directory(&output_directory).expect("the manifest validates");
}

/// An OptiQ-style vision tower: its own indexed file under `optiq/`.
fn add_vision_sidecar(model_directory: &Path) -> Vec<u8> {
    let vision_tensor = (
        "vision_tower.blocks.0.attn.proj.bias".to_owned(),
        "U8".to_owned(),
        vec![32],
        synthetic_tensor_bytes(250, 32),
    );
    std::fs::create_dir_all(model_directory.join("optiq")).expect("sidecar directory");
    let sidecar_path = model_directory.join("optiq/optiq_vision.safetensors");
    write_safetensors_shard(&sidecar_path, &[&vision_tensor]);
    let index_path = model_directory.join("model.safetensors.index.json");
    let mut index_document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path).expect("index reads"))
            .expect("index parses");
    index_document["weight_map"][&vision_tensor.0] =
        serde_json::Value::String("optiq/optiq_vision.safetensors".to_owned());
    std::fs::write(
        &index_path,
        serde_json::to_vec(&index_document).expect("index"),
    )
    .expect("index writes");
    std::fs::read(sidecar_path).expect("sidecar reads")
}

#[test]
fn should_carry_a_vision_sidecar_unchanged_and_keep_it_out_of_the_resident_bundle() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory.path().join("example/converted");
    write_synthetic_model(&source_directory);
    let sidecar_bytes = add_vision_sidecar(&source_directory);

    let conversion_summary =
        convert(&source_directory, &output_directory).expect("the conversion succeeds");

    assert_eq!(conversion_summary.resident_tensor_count, 3);
    let bundle_bytes =
        std::fs::read(output_directory.join(RESIDENT_BUNDLE_FILE_NAME)).expect("the bundle reads");
    let header_byte_count =
        u64::from_le_bytes(bundle_bytes[..8].try_into().expect("length prefix")) as usize;
    let header_document: serde_json::Value =
        serde_json::from_slice(&bundle_bytes[8..8 + header_byte_count]).expect("the header parses");
    assert!(
        header_document
            .get("vision_tower.blocks.0.attn.proj.bias")
            .is_none()
    );
    assert_eq!(
        std::fs::read(output_directory.join("optiq/optiq_vision.safetensors"))
            .expect("the sidecar travels with the Atlas"),
        sidecar_bytes
    );
}

#[test]
fn should_remove_its_staging_directory_when_a_write_fails() {
    let scratch_directory = tempfile::tempdir().expect("scratch");
    let source_directory = scratch_directory.path().join("example/model");
    let output_directory = scratch_directory
        .path()
        .join("example/model-astro-atlas-streaming");
    write_synthetic_model(&source_directory);
    // The header stays readable, so planning succeeds and the failure comes
    // only once payload bytes are copied into the staging directory.
    let truncated_shard_path = source_directory.join("model-00002.safetensors");
    let shard_bytes = std::fs::read(&truncated_shard_path).expect("the shard reads");
    let header_byte_count =
        u64::from_le_bytes(shard_bytes[..8].try_into().expect("length prefix")) as usize;
    std::fs::write(&truncated_shard_path, &shard_bytes[..8 + header_byte_count])
        .expect("the shard truncates");

    assert!(convert(&source_directory, &output_directory).is_err());
    assert!(!output_directory.exists());
    assert!(
        !scratch_directory
            .path()
            .join("example/.model-astro-atlas-streaming.atlas-partial")
            .exists()
    );
}
