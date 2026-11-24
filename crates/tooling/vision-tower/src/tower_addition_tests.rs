use std::fs;

use crate::base_matching::BaseMatchVerdict;
use crate::test_fixtures::{
    FixtureTensor, open_directory, seeded_values, write_bfloat16_file, write_shaped_bfloat16_file,
};

use super::*;

const LAYER_COUNT: u32 = 6;
const TEXT_CONFIG: &str = r#"{"image_token_id": 9, "text_config": {"hidden_size": 8},
    "quantization": {"bits": 4}}"#;
const TOWER_CONFIG: &str = r#"{"image_token_id": 9, "text_config": {"hidden_size": 8},
    "vision_config": {"depth": 1, "out_hidden_size": 8}}"#;
const PATCH_VALUE_COUNT: usize = 12;

#[derive(Default)]
struct RecordingProgress {
    stages: Vec<TowerAdditionStage>,
}

impl BaseMatchProgress for RecordingProgress {
    fn tensor_compared(&mut self, _completed_count: usize, _total_count: usize) {}
}

impl TowerAdditionProgress for RecordingProgress {
    fn stage_bytes(&mut self, stage: TowerAdditionStage, _completed: u64, _total: u64) {
        if !self.stages.contains(&stage) {
            self.stages.push(stage);
        }
    }
}

struct AdditionFixture {
    root: tempfile::TempDir,
}

impl AdditionFixture {
    /// A text-only model and an upstream-layout tower source sharing their text weights.
    fn new() -> Self {
        let fixture = Self {
            root: tempfile::tempdir().unwrap(),
        };
        fs::create_dir_all(fixture.text_directory().join("assets")).unwrap();
        write_text_weights(
            &fixture.text_directory(),
            "language_model.model.",
            LAYER_COUNT,
        );
        fs::write(fixture.text_directory().join("config.json"), TEXT_CONFIG).unwrap();
        fs::write(fixture.text_directory().join("tokenizer.json"), "{}").unwrap();
        fs::write(
            fixture.text_directory().join("README.md"),
            "the source card",
        )
        .unwrap();
        fs::write(fixture.text_directory().join("assets/logo.png"), "png").unwrap();
        fs::create_dir_all(fixture.tower_directory()).unwrap();
        write_upstream_checkpoint(&fixture.tower_directory());
        fs::write(fixture.tower_directory().join("config.json"), TOWER_CONFIG).unwrap();
        fs::write(
            fixture.tower_directory().join("preprocessor_config.json"),
            "{\"p\": 1}",
        )
        .unwrap();
        fixture
    }

    fn text_directory(&self) -> PathBuf {
        self.root.path().join("text")
    }

    fn tower_directory(&self) -> PathBuf {
        self.root.path().join("tower")
    }

    fn output_directory(&self) -> PathBuf {
        self.root.path().join("output")
    }

    async fn add(&self) -> Result<TowerAdditionReport, VisionTowerError> {
        let text_source = open_directory(&self.text_directory()).await;
        let tower_source = open_directory(&self.tower_directory()).await;
        let output_directory = self.output_directory();
        add_tower(
            TowerAdditionRequest {
                text_source: &text_source,
                tower_source: &tower_source,
                output_directory: &output_directory,
            },
            &mut RecordingProgress::default(),
        )
        .await
    }
}

fn text_values(layer_index: u32) -> Vec<f32> {
    seeded_values(layer_index, 6)
}

fn write_text_weights(directory: &Path, prefix: &str, shared_layers: u32) {
    let mut layer_names = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        layer_names.push(format!("{prefix}layers.{layer_index}.conv1d.weight"));
    }
    let mut tensors = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        let seed = if layer_index < shared_layers {
            layer_index
        } else {
            layer_index + 500
        };
        tensors.push(FixtureTensor {
            name: &layer_names[layer_index as usize],
            float_values: seeded_values(seed, 6),
        });
    }
    write_bfloat16_file(&directory.join("model.safetensors"), &tensors);
}

fn patch_values() -> Vec<f32> {
    let mut float_values = Vec::with_capacity(PATCH_VALUE_COUNT);
    for value_index in 0..PATCH_VALUE_COUNT {
        float_values.push(value_index as f32);
    }
    float_values
}

/// One real shard holding text tensors and a tower, named as upstream names them.
fn write_upstream_checkpoint(directory: &Path) {
    let mut names = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        names.push(format!(
            "model.language_model.layers.{layer_index}.conv1d.weight"
        ));
    }
    let mut values = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        values.push(text_values(layer_index));
    }
    let patch = patch_values();
    let block_values = seeded_values(70, 4);
    let mut tensors: Vec<(&str, Vec<u64>, &[f32])> = Vec::new();
    for layer_index in 0..LAYER_COUNT as usize {
        tensors.push((&names[layer_index], vec![6], &values[layer_index]));
    }
    // [out 2, in 3, depth 2, height 1, width 1]
    tensors.push((
        "model.visual.patch_embed.proj.weight",
        vec![2, 3, 2, 1, 1],
        &patch,
    ));
    tensors.push((
        "model.visual.blocks.0.attn.proj.weight",
        vec![4],
        &block_values,
    ));
    write_shaped_bfloat16_file(
        &directory.join("model-00001-of-00001.safetensors"),
        &tensors,
    );
}

fn read_json(path: PathBuf) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[tokio::test]
async fn should_build_a_model_loaders_can_find_the_tower_in() {
    let fixture = AdditionFixture::new();
    let report = fixture.add().await.unwrap();
    assert_eq!(report.base_match.verdict, BaseMatchVerdict::SameBase);
    assert_eq!(report.tower_tensor_count, 2);
    assert_eq!(report.tower_byte_count, 32);

    let output = fixture.output_directory();
    let config = read_json(output.join("config.json"));
    assert_eq!(config["vision_config"]["depth"], 1);
    assert_eq!(config["quantization"]["bits"], 4, "text settings survive");
    let index = read_json(output.join("model.safetensors.index.json"));
    assert_eq!(
        index["weight_map"]["vision_tower.patch_embed.proj.weight"],
        TOWER_FILE_RELATIVE_PATH
    );
    assert_eq!(
        index["weight_map"]["language_model.model.layers.0.conv1d.weight"],
        "model.safetensors"
    );

    let inspection = inspect_artifact(&open_directory(&output).await)
        .await
        .unwrap();
    assert_eq!(inspection.status, TowerStatus::Present);
    assert_eq!(inspection.tower.unwrap().name_prefixes, ["vision_tower"]);
}

#[tokio::test]
async fn should_keep_the_text_weights_byte_for_byte() {
    let fixture = AdditionFixture::new();
    fixture.add().await.unwrap();
    assert_eq!(
        fs::read(fixture.output_directory().join("model.safetensors")).unwrap(),
        fs::read(fixture.text_directory().join("model.safetensors")).unwrap()
    );
}

#[tokio::test]
async fn should_carry_settings_over_and_leave_the_source_card_behind() {
    let fixture = AdditionFixture::new();
    let report = fixture.add().await.unwrap();
    let output = fixture.output_directory();
    assert!(output.join("tokenizer.json").is_file());
    assert_eq!(
        fs::read(output.join("preprocessor_config.json")).unwrap(),
        b"{\"p\": 1}"
    );
    assert!(!output.join("README.md").exists());
    assert!(!output.join("assets").exists());
    assert_eq!(
        report.missing_image_settings,
        ["processor_config.json", "video_preprocessor_config.json"]
    );
}

#[tokio::test]
async fn should_reorder_the_patch_embedding_and_keep_other_tensors_exact() {
    let fixture = AdditionFixture::new();
    fixture.add().await.unwrap();
    let source = open_directory(&fixture.output_directory()).await;
    let inventory = read_weight_inventory(&source).await.unwrap();
    let patch = &inventory.tensors["vision_tower.patch_embed.proj.weight"];
    assert_eq!(patch.entry.shape, [2, 2, 1, 1, 3]);
    let patch_bytes = source
        .read_byte_range(
            &patch.relative_path,
            patch.entry.data_start,
            patch.entry.byte_count,
        )
        .await
        .unwrap();
    let mut decoded = Vec::new();
    for element_bytes in patch_bytes.as_chunks::<2>().0 {
        decoded.push(f32::from_bits(
            u32::from(u16::from_le_bytes(*element_bytes)) << 16,
        ));
    }
    // Source order is [out][in][depth]: for output 0, values 0..6 are in 0 depth 0,
    // in 0 depth 1, in 1 depth 0, and so on. Channels last groups by depth first.
    assert_eq!(
        decoded,
        [0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 6.0, 8.0, 10.0, 7.0, 9.0, 11.0]
    );
}

#[tokio::test]
async fn should_write_a_record_with_hashes_and_no_local_paths() {
    let fixture = AdditionFixture::new();
    fixture.add().await.unwrap();
    let output = fixture.output_directory();
    let record_text = fs::read_to_string(output.join(BUILD_RECORD_FILE_NAME)).unwrap();
    let record: Value = serde_json::from_str(&record_text).unwrap();
    assert_eq!(record["base_match"]["verdict"], "same_base");
    assert_eq!(record["base_match"]["shared_tensor_count"], 6);
    let root_text = fixture.root.path().to_str().unwrap();
    assert!(
        !record_text.contains(root_text),
        "the record is published: {record_text}"
    );
    let mut was_weight_file_recorded = false;
    for file_record in record["files"].as_array().unwrap() {
        if file_record["path"] == "model.safetensors" {
            let (digest, _) = hash_file(&output.join("model.safetensors")).unwrap();
            assert_eq!(file_record["sha256"], digest.as_str());
            was_weight_file_recorded = true;
        }
    }
    assert!(was_weight_file_recorded);
}

#[tokio::test]
async fn should_build_the_same_files_every_time() {
    let fixture = AdditionFixture::new();
    fixture.add().await.unwrap();
    let first_record = fs::read(fixture.output_directory().join(BUILD_RECORD_FILE_NAME)).unwrap();
    fs::remove_dir_all(fixture.output_directory()).unwrap();
    fixture.add().await.unwrap();
    let second_record = fs::read(fixture.output_directory().join(BUILD_RECORD_FILE_NAME)).unwrap();
    assert_eq!(first_record, second_record);
}

#[tokio::test]
async fn should_treat_a_trailing_slash_on_the_output_as_the_same_folder() {
    let fixture = AdditionFixture::new();
    let text_source = open_directory(&fixture.text_directory()).await;
    let tower_source = open_directory(&fixture.tower_directory()).await;
    let output_with_slash = PathBuf::from(format!("{}/", fixture.output_directory().display()));
    add_tower(
        TowerAdditionRequest {
            text_source: &text_source,
            tower_source: &tower_source,
            output_directory: &output_with_slash,
        },
        &mut RecordingProgress::default(),
    )
    .await
    .unwrap();
    assert!(fixture.output_directory().join("config.json").is_file());
    assert!(!fixture.output_directory().join(".incomplete").exists());
    assert!(!fixture.root.path().join("output.incomplete").exists());
}

#[test]
fn should_refuse_an_output_path_with_no_name() {
    assert!(matches!(
        incomplete_directory_for(Path::new("folder/..")),
        Err(VisionTowerError::OutputAlreadyExists { .. })
    ));
}

#[tokio::test]
async fn should_never_overwrite_an_existing_output() {
    let fixture = AdditionFixture::new();
    fs::create_dir(fixture.output_directory()).unwrap();
    fs::write(fixture.output_directory().join("keep.txt"), "mine").unwrap();
    assert!(matches!(
        fixture.add().await,
        Err(VisionTowerError::OutputAlreadyExists { .. })
    ));
    assert_eq!(
        fs::read(fixture.output_directory().join("keep.txt")).unwrap(),
        b"mine"
    );
}

#[tokio::test]
async fn should_stop_at_a_leftover_from_an_interrupted_run() {
    let fixture = AdditionFixture::new();
    fs::create_dir(fixture.root.path().join("output.incomplete")).unwrap();
    assert!(matches!(
        fixture.add().await,
        Err(VisionTowerError::IncompleteOutputPresent { .. })
    ));
}

#[tokio::test]
async fn should_refuse_a_tower_with_no_shared_weights_and_write_nothing() {
    let fixture = AdditionFixture::new();
    write_text_weights(&fixture.text_directory(), "language_model.model.", 1);
    let outcome = fixture.add().await;
    assert!(matches!(
        outcome,
        Err(VisionTowerError::AdditionRefused { .. })
    ));
    assert!(!fixture.output_directory().exists());
    assert!(!fixture.root.path().join("output.incomplete").exists());
}

#[tokio::test]
async fn should_refuse_a_text_model_that_already_has_a_tower() {
    let fixture = AdditionFixture::new();
    write_bfloat16_file(
        &fixture.text_directory().join("optiq/tower.safetensors"),
        &[FixtureTensor {
            name: "vision_tower.a.weight",
            float_values: seeded_values(9, 2),
        }],
    );
    let Err(VisionTowerError::AdditionRefused { reason }) = fixture.add().await else {
        panic!("the addition should be refused");
    };
    assert!(reason.contains("already has"), "{reason}");
}

#[tokio::test]
async fn should_refuse_a_quantized_tower() {
    let fixture = AdditionFixture::new();
    let mut names = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        names.push(format!(
            "model.language_model.layers.{layer_index}.conv1d.weight"
        ));
    }
    let mut tensors = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        tensors.push(FixtureTensor {
            name: &names[layer_index as usize],
            float_values: text_values(layer_index),
        });
    }
    tensors.push(FixtureTensor {
        name: "model.visual.blocks.0.attn.proj.scales",
        float_values: seeded_values(71, 2),
    });
    write_bfloat16_file(
        &fixture
            .tower_directory()
            .join("model-00001-of-00001.safetensors"),
        &tensors,
    );
    let Err(VisionTowerError::AdditionRefused { reason }) = fixture.add().await else {
        panic!("the addition should be refused");
    };
    assert!(reason.contains("quantized"), "{reason}");
}

#[tokio::test]
async fn should_remove_its_half_built_folder_when_the_tower_cannot_be_read() {
    let fixture = AdditionFixture::new();
    let text_source = open_directory(&fixture.text_directory()).await;
    let tower_source = open_directory(&fixture.tower_directory()).await;
    // Cut the tower's data off after its source was opened and its match has run:
    // the header stays readable, so this fails partway through the build.
    let shard = fixture
        .tower_directory()
        .join("model-00001-of-00001.safetensors");
    let shard_bytes = fs::read(&shard).unwrap();
    fs::write(&shard, &shard_bytes[..shard_bytes.len() - 4]).unwrap();
    let output_directory = fixture.output_directory();
    let outcome = add_tower(
        TowerAdditionRequest {
            text_source: &text_source,
            tower_source: &tower_source,
            output_directory: &output_directory,
        },
        &mut RecordingProgress::default(),
    )
    .await;
    assert!(outcome.is_err());
    assert!(!output_directory.exists());
    assert!(!fixture.root.path().join("output.incomplete").exists());
}

#[tokio::test]
async fn should_report_progress_for_the_tower_and_the_hashes() {
    let fixture = AdditionFixture::new();
    let text_source = open_directory(&fixture.text_directory()).await;
    let tower_source = open_directory(&fixture.tower_directory()).await;
    let output_directory = fixture.output_directory();
    let mut progress = RecordingProgress::default();
    add_tower(
        TowerAdditionRequest {
            text_source: &text_source,
            tower_source: &tower_source,
            output_directory: &output_directory,
        },
        &mut progress,
    )
    .await
    .unwrap();
    assert_eq!(
        progress.stages,
        [
            TowerAdditionStage::FetchingTower,
            TowerAdditionStage::HashingFiles
        ]
    );
}
