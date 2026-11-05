//! The `astronomical vision` journeys against small real safetensors files in a
//! temporary directory. No daemon, no GPU, no network; the journeys against real
//! Hugging Face repositories live in the vision tower crate.

use std::fs;
use std::path::Path;
use std::time::Duration;

use astronomical_cli::{
    VisionAddArguments, VisionDependencies, VisionError, VisionInspectArguments,
    VisionMatchArguments, run_vision_add, run_vision_inspect, run_vision_match,
};
use astronomical_vision_tower::{BaseMatchVerdict, TowerStatus};

use crate::integration::support::environment::fresh_test_directory;

const CONFIG_WITH_VISION: &str = r#"{"image_token_id": 9, "text_config": {"hidden_size": 8}, "vision_config": {"depth": 1, "out_hidden_size": 8}}"#;
const TEXT_LAYER_COUNT: u16 = 6;

/// Writes one bfloat16 file whose tensors are named by `tensor_names`, with values
/// that vary inside a tensor and differ between tensors.
fn write_weights(file_path: &Path, tensor_names: &[String]) {
    let mut header_entries = serde_json::Map::new();
    let mut tensor_bytes = Vec::new();
    for (tensor_index, tensor_name) in tensor_names.iter().enumerate() {
        let start_offset = tensor_bytes.len();
        for value_index in 0..4_u16 {
            let raw_value = 0x3f80_u16 + tensor_index as u16 * 16 + value_index;
            tensor_bytes.extend_from_slice(&raw_value.to_le_bytes());
        }
        header_entries.insert(
            tensor_name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": [4],
                               "data_offsets": [start_offset, tensor_bytes.len()]}),
        );
    }
    let header_bytes = serde_json::to_vec(&header_entries).expect("the header serializes");
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(&tensor_bytes);
    fs::create_dir_all(file_path.parent().expect("the file has a folder"))
        .expect("the fixture folder is writable");
    fs::write(file_path, file_bytes).expect("the fixture file is writable");
}

fn text_tensor_names(prefix: &str) -> Vec<String> {
    let mut tensor_names = Vec::new();
    for layer_index in 0..TEXT_LAYER_COUNT {
        tensor_names.push(format!("{prefix}layers.{layer_index}.conv1d.weight"));
    }
    tensor_names
}

fn write_text_only_model(model_directory: &Path) {
    write_weights(
        &model_directory.join("model.safetensors"),
        &text_tensor_names("language_model.model."),
    );
    fs::write(
        model_directory.join("config.json"),
        r#"{"image_token_id": 9, "hidden_size": 8}"#,
    )
    .expect("the config is writable");
}

fn write_model_with_tower(model_directory: &Path) {
    write_weights(
        &model_directory.join("model.safetensors"),
        &text_tensor_names("model.language_model."),
    );
    write_weights(
        &model_directory.join("optiq/optiq_vision.safetensors"),
        &["vision_tower.blocks.0.attn.proj.weight".to_owned()],
    );
    fs::write(model_directory.join("config.json"), CONFIG_WITH_VISION)
        .expect("the config is writable");
}

struct CapturedRun<Outcome> {
    outcome: Result<Outcome, VisionError>,
    stdout: String,
}

fn inspect(model: &Path, render_json: bool) -> CapturedRun<TowerStatus> {
    let arguments = VisionInspectArguments {
        model: model.display().to_string(),
        render_json,
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = VisionDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let outcome =
        current_thread_runtime().block_on(run_vision_inspect(&arguments, &mut dependencies));
    CapturedRun {
        outcome,
        stdout: String::from_utf8(stdout).expect("the report is text"),
    }
}

fn match_models(
    text_model: &Path,
    tower_model: &Path,
    render_json: bool,
) -> CapturedRun<BaseMatchVerdict> {
    let arguments = VisionMatchArguments {
        text_model: text_model.display().to_string(),
        tower_model: tower_model.display().to_string(),
        render_json,
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = VisionDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let outcome =
        current_thread_runtime().block_on(run_vision_match(&arguments, &mut dependencies));
    CapturedRun {
        outcome,
        stdout: String::from_utf8(stdout).expect("the report is text"),
    }
}

fn add_tower_to(
    text_model: &Path,
    tower_model: &Path,
    output_directory: &Path,
) -> CapturedRun<u64> {
    let arguments = VisionAddArguments {
        text_model: text_model.display().to_string(),
        tower_model: tower_model.display().to_string(),
        output_directory: output_directory.to_path_buf(),
        render_json: false,
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = VisionDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::from_secs(5),
    };
    let outcome = current_thread_runtime().block_on(run_vision_add(&arguments, &mut dependencies));
    CapturedRun {
        outcome: match outcome {
            Ok(report) => Ok(report.tower_tensor_count),
            Err(vision_error) => Err(vision_error),
        },
        stdout: String::from_utf8(stdout).expect("the report is text"),
    }
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the runtime builds")
}

#[test]
fn should_say_a_text_only_model_has_no_tower_and_why() {
    let test_directory = fresh_test_directory("vision", "text-only");
    write_text_only_model(&test_directory);
    let run = inspect(&test_directory, false);
    assert_eq!(
        run.outcome.expect("inspection succeeds"),
        TowerStatus::AbsentWithImageTokens
    );
    assert!(run.stdout.contains("Vision tower: none"), "{}", run.stdout);
    assert!(run.stdout.contains("image tokens"), "{}", run.stdout);
    assert!(run.stdout.contains("6 tensors"), "{}", run.stdout);
}

#[test]
fn should_describe_a_tower_kept_in_a_sidecar() {
    let test_directory = fresh_test_directory("vision", "sidecar");
    write_model_with_tower(&test_directory);
    let run = inspect(&test_directory, false);
    assert_eq!(
        run.outcome.expect("inspection succeeds"),
        TowerStatus::Present
    );
    assert!(run.stdout.contains("Vision tower: yes"), "{}", run.stdout);
    assert!(
        run.stdout.contains("optiq/optiq_vision.safetensors"),
        "{}",
        run.stdout
    );
    assert!(
        run.stdout.contains("Output width: 8, text width: 8"),
        "{}",
        run.stdout
    );
}

#[test]
fn should_render_an_inspection_as_one_json_document() {
    let test_directory = fresh_test_directory("vision", "json");
    write_model_with_tower(&test_directory);
    let run = inspect(&test_directory, true);
    run.outcome.expect("inspection succeeds");
    let document: serde_json::Value =
        serde_json::from_str(&run.stdout).expect("standard output is one JSON document");
    assert_eq!(document["status"], "present");
    assert_eq!(document["tower"]["tensor_count"], 1);
}

#[test]
fn should_prove_a_shared_base_across_naming_conventions() {
    let test_directory = fresh_test_directory("vision", "same-base");
    let text_model = test_directory.join("text");
    let tower_model = test_directory.join("tower");
    write_text_only_model(&created_folder(&text_model));
    write_model_with_tower(&created_folder(&tower_model));
    let run = match_models(&text_model, &tower_model, false);
    assert_eq!(
        run.outcome.expect("the match succeeds"),
        BaseMatchVerdict::SameBase
    );
    assert!(run.stdout.contains("Verdict: same base"), "{}", run.stdout);
    assert!(run.stdout.contains("6 identical"), "{}", run.stdout);
}

#[test]
fn should_refuse_a_source_that_has_no_tower() {
    let test_directory = fresh_test_directory("vision", "no-tower-source");
    let text_model = test_directory.join("text");
    let other_model = test_directory.join("other");
    write_text_only_model(&created_folder(&text_model));
    write_text_only_model(&created_folder(&other_model));
    let run = match_models(&text_model, &other_model, false);
    assert_eq!(
        run.outcome.expect("the match succeeds"),
        BaseMatchVerdict::Incompatible
    );
    assert!(run.stdout.contains("no vision tower"), "{}", run.stdout);
}

#[test]
fn should_fail_clearly_on_something_that_is_not_a_model_reference() {
    let test_directory = fresh_test_directory("vision", "bad-reference");
    let run = inspect(&test_directory.join("no/such/folder/here"), false);
    let failure = run.outcome.expect_err("the reference is refused");
    assert!(
        failure
            .to_string()
            .contains("neither a local model directory"),
        "{failure}"
    );
}

#[test]
fn should_add_a_tower_and_report_what_was_built() {
    let test_directory = fresh_test_directory("vision", "add");
    let text_model = created_folder(&test_directory.join("text"));
    let tower_model = created_folder(&test_directory.join("tower"));
    let output_directory = test_directory.join("output");
    write_text_only_model(&text_model);
    write_model_with_tower(&tower_model);

    let run = add_tower_to(&text_model, &tower_model, &output_directory);

    assert_eq!(run.outcome.expect("the add succeeds"), 1);
    assert!(
        run.stdout.contains("Text weights: unchanged"),
        "{}",
        run.stdout
    );
    assert!(run.stdout.contains("Verdict: same base"), "{}", run.stdout);
    let built_model = inspect(&output_directory, false);
    assert_eq!(
        built_model.outcome.expect("inspection succeeds"),
        TowerStatus::Present
    );
}

#[test]
fn should_report_progress_with_an_estimate_on_standard_error() {
    let test_directory = fresh_test_directory("vision", "add-progress");
    let text_model = created_folder(&test_directory.join("text"));
    let tower_model = created_folder(&test_directory.join("tower"));
    write_text_only_model(&text_model);
    write_model_with_tower(&tower_model);
    let arguments = VisionAddArguments {
        text_model: text_model.display().to_string(),
        tower_model: tower_model.display().to_string(),
        output_directory: test_directory.join("output"),
        render_json: false,
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut dependencies = VisionDependencies {
        stdout: &mut stdout,
        stderr: &mut stderr,
        progress_interval: Duration::ZERO,
    };
    current_thread_runtime()
        .block_on(run_vision_add(&arguments, &mut dependencies))
        .expect("the add succeeds");
    let progress_text = String::from_utf8(stderr).expect("progress is text");
    for stage_name in ["[compare]", "[tower]", "[hash]"] {
        assert!(progress_text.contains(stage_name), "{progress_text}");
    }
    assert!(progress_text.contains("eta"), "{progress_text}");
}

#[test]
fn should_write_nothing_when_the_tower_source_has_no_tower() {
    let test_directory = fresh_test_directory("vision", "add-refused");
    let text_model = created_folder(&test_directory.join("text"));
    let other_model = created_folder(&test_directory.join("other"));
    let output_directory = test_directory.join("output");
    write_text_only_model(&text_model);
    write_text_only_model(&other_model);

    let run = add_tower_to(&text_model, &other_model, &output_directory);

    let failure = run.outcome.expect_err("the add is refused");
    assert!(failure.to_string().contains("was not added"), "{failure}");
    assert!(!output_directory.exists());
}

fn created_folder(model_directory: &Path) -> std::path::PathBuf {
    fs::create_dir_all(model_directory).expect("the model folder is writable");
    model_directory.to_path_buf()
}
