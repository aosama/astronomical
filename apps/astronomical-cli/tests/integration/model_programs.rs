//! The `astronomical models programs` report. No daemon, no GPU, no network.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use astronomical_cli::{ModelProgramsReport, render_model_programs};
use astronomical_config::list_neural_engine_program_sets;

use crate::integration::support::environment::fresh_test_directory;

fn write_set(root: &Path, model_id: &str, is_finished: bool) {
    let set_directory = root
        .join(model_id)
        .join("0123456789abcdef")
        .join("mlp-v1-960-of-2048");
    let program = set_directory.join("layer-0.mlmodelc");
    fs::create_dir_all(&program).expect("the fixture folder is writable");
    fs::write(program.join("weight.bin"), vec![0_u8; 2_000_000]).expect("the weights write");
    if is_finished {
        fs::write(set_directory.join("programs.json"), "{}").expect("the manifest writes");
    }
}

fn report_text(root: &Path, installed_model_ids: Option<&BTreeSet<String>>) -> String {
    let program_sets = list_neural_engine_program_sets(root);
    let mut output = Vec::new();
    render_model_programs(
        &ModelProgramsReport {
            program_directory: root,
            program_sets: &program_sets,
            installed_model_ids,
        },
        &mut output,
    )
    .expect("the report writes");
    String::from_utf8(output).expect("the report is text")
}

#[test]
fn should_show_the_declared_folder_even_with_nothing_built() {
    let root = fresh_test_directory("model-programs", "empty");
    let text = report_text(&root, Some(&BTreeSet::new()));
    assert!(text.contains(&root.display().to_string()), "{text}");
    assert!(text.contains("None yet"), "{text}");
}

#[test]
fn should_list_each_set_with_its_size_and_a_total() {
    let root = fresh_test_directory("model-programs", "listed");
    write_set(&root, "installed-model", true);
    let installed = BTreeSet::from(["installed-model".to_owned()]);
    let text = report_text(&root, Some(&installed));
    assert!(
        text.contains("installed-model (0123456789ab) mlp-v1-960-of-2048: 1 programs"),
        "{text}"
    );
    assert!(text.contains("Total:"), "{text}");
    assert!(!text.contains("[model not found"), "{text}");
    assert!(!text.contains("[incomplete]"), "{text}");
}

#[test]
fn should_flag_a_set_whose_model_is_gone_or_that_never_finished() {
    let root = fresh_test_directory("model-programs", "flagged");
    write_set(&root, "removed-model", false);
    let text = report_text(&root, Some(&BTreeSet::new()));
    assert!(text.contains("[model not found]"), "{text}");
    assert!(text.contains("[incomplete]"), "{text}");
}

#[test]
fn should_not_call_anything_missing_when_discovery_failed() {
    let root = fresh_test_directory("model-programs", "unknown");
    write_set(&root, "some-model", true);
    let text = report_text(&root, None);
    assert!(!text.contains("[model not found"), "{text}");
}
