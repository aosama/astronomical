use std::fs;
use std::path::Path;

use crate::test_fixtures::{FixtureTensor, open_directory, seeded_values, write_bfloat16_file};

use super::*;

const COMPATIBLE_CONFIG: &str = r#"{"image_token_id": 9, "video_token_id": 10,
    "vision_start_token_id": 11, "vision_end_token_id": 12,
    "text_config": {"hidden_size": 8},
    "vision_config": {"depth": 1, "out_hidden_size": 8}}"#;
const LAYER_COUNT: u32 = 8;

#[derive(Default)]
struct CountingProgress {
    last_report: Option<(usize, usize)>,
}

impl BaseMatchProgress for CountingProgress {
    fn tensor_compared(&mut self, completed_count: usize, total_count: usize) {
        self.last_report = Some((completed_count, total_count));
    }
}

fn layer_name(prefix: &str, layer_index: u32) -> String {
    format!("{prefix}layers.{layer_index}.linear_attn.conv1d.weight")
}

/// Writes text layers under `prefix`; layers at or past `edited_from` use other
/// weights, as a separately trained or edited model would.
fn write_text_weights(directory: &Path, prefix: &str, edited_from: u32) {
    let mut layer_names = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        layer_names.push(layer_name(prefix, layer_index));
    }
    let mut tensors = Vec::new();
    for layer_index in 0..LAYER_COUNT {
        let seed = if layer_index >= edited_from {
            layer_index + 100
        } else {
            layer_index
        };
        tensors.push(FixtureTensor {
            name: &layer_names[layer_index as usize],
            float_values: seeded_values(seed, 6),
        });
    }
    write_bfloat16_file(&directory.join("model.safetensors"), &tensors);
}

fn write_tower(directory: &Path) {
    write_bfloat16_file(
        &directory.join("optiq/optiq_vision.safetensors"),
        &[FixtureTensor {
            name: "vision_tower.blocks.0.attn.proj.weight",
            float_values: seeded_values(50, 4),
        }],
    );
}

fn write_config(directory: &Path, config_text: &str) {
    fs::write(directory.join("config.json"), config_text).unwrap();
}

async fn match_directories(text_directory: &Path, tower_directory: &Path) -> BaseMatchReport {
    let text_source = open_directory(text_directory).await;
    let tower_source = open_directory(tower_directory).await;
    match_tower_to_text(
        &text_source,
        &tower_source,
        &mut CountingProgress::default(),
    )
    .await
    .unwrap()
}

struct MatchFixture {
    text_directory: tempfile::TempDir,
    tower_directory: tempfile::TempDir,
}

fn match_fixture(edited_from: u32) -> MatchFixture {
    let fixture = MatchFixture {
        text_directory: tempfile::tempdir().unwrap(),
        tower_directory: tempfile::tempdir().unwrap(),
    };
    // Two naming conventions, as an MLX conversion and its upstream really differ.
    write_text_weights(
        fixture.text_directory.path(),
        "language_model.model.",
        edited_from,
    );
    write_text_weights(
        fixture.tower_directory.path(),
        "model.language_model.",
        LAYER_COUNT,
    );
    write_tower(fixture.tower_directory.path());
    write_config(fixture.text_directory.path(), COMPATIBLE_CONFIG);
    write_config(fixture.tower_directory.path(), COMPATIBLE_CONFIG);
    fixture
}

async fn verdict_of(fixture: &MatchFixture) -> BaseMatchReport {
    match_directories(
        fixture.text_directory.path(),
        fixture.tower_directory.path(),
    )
    .await
}

#[tokio::test]
async fn should_find_the_same_base_across_naming_conventions() {
    let report = verdict_of(&match_fixture(LAYER_COUNT)).await;
    assert_eq!(report.verdict, BaseMatchVerdict::SameBase);
    assert_eq!(report.comparable_tensor_count, u64::from(LAYER_COUNT));
    assert_eq!(report.compared_tensors.len(), LAYER_COUNT as usize);
    assert!(report.incompatibilities.is_empty());
    assert!(report.unchecked.is_empty());
    assert!(!report.text_already_has_tower);
    assert!(report.verdict.allows_adding());
}

#[tokio::test]
async fn should_call_a_partly_edited_model_derived() {
    let report = verdict_of(&match_fixture(LAYER_COUNT - 3)).await;
    assert_eq!(report.verdict, BaseMatchVerdict::DerivedFromBase);
    assert!(report.verdict.allows_adding());
}

#[tokio::test]
async fn should_keep_derived_at_exactly_one_shared_in_four() {
    let report = verdict_of(&match_fixture(LAYER_COUNT / 4)).await;
    assert_eq!(report.verdict, BaseMatchVerdict::DerivedFromBase);
}

#[tokio::test]
async fn should_find_no_shared_weights_when_almost_nothing_agrees() {
    let report = verdict_of(&match_fixture(1)).await;
    assert_eq!(report.verdict, BaseMatchVerdict::NoSharedWeights);
    assert!(!report.verdict.allows_adding());
}

#[tokio::test]
async fn should_refuse_a_tower_whose_output_width_does_not_fit() {
    let fixture = match_fixture(LAYER_COUNT);
    write_config(
        fixture.tower_directory.path(),
        r#"{"vision_config": {"out_hidden_size": 16}}"#,
    );
    write_config(fixture.text_directory.path(), r#"{"hidden_size": 8}"#);
    let report = verdict_of(&fixture).await;
    assert_eq!(report.verdict, BaseMatchVerdict::Incompatible);
    assert!(report.incompatibilities[0].contains("16"));
    assert!(
        report.compared_tensors.is_empty(),
        "no tensors are read after a refusal"
    );
}

#[tokio::test]
async fn should_refuse_differing_image_tokens() {
    let fixture = match_fixture(LAYER_COUNT);
    write_config(
        fixture.tower_directory.path(),
        r#"{"image_token_id": 77, "vision_config": {"out_hidden_size": 8}}"#,
    );
    let report = verdict_of(&fixture).await;
    assert_eq!(report.verdict, BaseMatchVerdict::Incompatible);
    assert!(report.incompatibilities[0].contains("image_token_id"));
}

#[tokio::test]
async fn should_refuse_a_source_with_no_tower() {
    let fixture = match_fixture(LAYER_COUNT);
    fs::remove_dir_all(fixture.tower_directory.path().join("optiq")).unwrap();
    let report = verdict_of(&fixture).await;
    assert_eq!(report.verdict, BaseMatchVerdict::Incompatible);
    assert!(report.incompatibilities[0].contains("no vision tower"));
}

#[tokio::test]
async fn should_list_what_the_configs_did_not_state_rather_than_assume_it() {
    let fixture = match_fixture(LAYER_COUNT);
    write_config(fixture.text_directory.path(), "{}");
    let report = verdict_of(&fixture).await;
    assert_eq!(report.verdict, BaseMatchVerdict::SameBase);
    assert!(report.unchecked.contains(&"tower output width".to_owned()));
    assert!(report.unchecked.contains(&"image_token_id".to_owned()));
}

#[tokio::test]
async fn should_be_inconclusive_with_too_little_to_compare() {
    let text_directory = tempfile::tempdir().unwrap();
    let tower_directory = tempfile::tempdir().unwrap();
    write_tower(tower_directory.path());
    write_config(text_directory.path(), COMPATIBLE_CONFIG);
    write_config(tower_directory.path(), COMPATIBLE_CONFIG);
    let report = match_directories(text_directory.path(), tower_directory.path()).await;
    assert_eq!(report.verdict, BaseMatchVerdict::Inconclusive);
    assert!(!report.verdict.allows_adding());
}

#[tokio::test]
async fn should_not_compare_the_tower_or_quantized_weights() {
    let text_directory = tempfile::tempdir().unwrap();
    write_bfloat16_file(
        &text_directory.path().join("model.safetensors"),
        &[
            FixtureTensor {
                name: "layers.0.up.weight",
                float_values: seeded_values(1, 4),
            },
            FixtureTensor {
                name: "layers.0.up.scales",
                float_values: seeded_values(2, 4),
            },
            FixtureTensor {
                name: "layers.0.up.biases",
                float_values: seeded_values(3, 4),
            },
            FixtureTensor {
                name: "vision_tower.a.weight",
                float_values: seeded_values(4, 4),
            },
            FixtureTensor {
                name: "layers.0.norm.weight",
                float_values: seeded_values(5, 4),
            },
        ],
    );
    let inventory = WeightInventory::default();
    let text_source = open_directory(text_directory.path()).await;
    let text_inventory = read_weight_inventory(&text_source).await.unwrap();
    assert!(comparable_tensor_pairs(&text_inventory, &inventory).is_empty());
    let comparable = comparable_tensors_by_canonical_name(&text_inventory);
    assert_eq!(
        comparable.keys().copied().collect::<Vec<_>>(),
        ["layers.0.norm.weight"]
    );
}

#[tokio::test]
async fn should_report_progress_through_every_comparison() {
    let fixture = match_fixture(LAYER_COUNT);
    let text_source = open_directory(fixture.text_directory.path()).await;
    let tower_source = open_directory(fixture.tower_directory.path()).await;
    let mut progress = CountingProgress::default();
    match_tower_to_text(&text_source, &tower_source, &mut progress)
        .await
        .unwrap();
    assert_eq!(
        progress.last_report,
        Some((LAYER_COUNT as usize, LAYER_COUNT as usize))
    );
}

#[test]
fn should_space_a_selection_across_the_whole_model() {
    let located_tensor = LocatedTensor {
        relative_path: String::new(),
        entry: crate::safetensors_header::TensorEntry {
            data_type: "BF16".to_owned(),
            shape: Vec::new(),
            data_start: 0,
            byte_count: 0,
        },
    };
    let mut names = Vec::new();
    for name_index in 0..100 {
        names.push(name_index.to_string());
    }
    let mut pairs = Vec::new();
    for name in &names {
        pairs.push(TensorPair {
            canonical_name: name,
            text_tensor: &located_tensor,
            tower_side_tensor: &located_tensor,
        });
    }
    let selected = evenly_spaced_pairs(&pairs, 4);
    let mut selected_names = Vec::new();
    for selected_pair in &selected {
        selected_names.push(selected_pair.canonical_name);
    }
    assert_eq!(selected_names, ["0", "25", "50", "75"]);
    assert_eq!(evenly_spaced_pairs(&pairs[..2], 4).len(), 2);
    assert!(evenly_spaced_pairs(&pairs[..0], 4).is_empty());
}
