use super::{MemoryShortfall, ProblemReport};
use crate::problem_reports::{CatalogModelName, MacDescription, ProblemKind};

fn this_test_mac() -> MacDescription {
    MacDescription {
        chip_name: "Apple M5 Pro".to_owned(),
        memory_gb: 48,
        macos_version: "27.2".to_owned(),
    }
}

fn catalog_ids() -> Vec<String> {
    vec![
        "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit".to_owned(),
        "mlx-community/Qwen3.5-2B-4bit".to_owned(),
    ]
}

#[test]
fn should_name_a_catalog_model_by_its_id_or_its_folder_name() {
    assert_eq!(
        CatalogModelName::from_model_id("mlx-community/Qwen3.5-2B-4bit", &catalog_ids()),
        CatalogModelName::Listed("mlx-community/Qwen3.5-2B-4bit".to_owned())
    );
    assert_eq!(
        CatalogModelName::from_model_id("Qwen3.5-2B-4bit", &catalog_ids()),
        CatalogModelName::Listed("mlx-community/Qwen3.5-2B-4bit".to_owned())
    );
}

#[test]
fn should_never_name_a_model_the_catalog_does_not_list() {
    for model_id in ["my-own-finetune", "someone/private-model", "", "Qwen3.5-2B"] {
        let model_name = CatalogModelName::from_model_id(model_id, &catalog_ids());
        assert_eq!(model_name, CatalogModelName::NotInCatalog, "{model_id:?}");
        assert_eq!(model_name.display_text(), "a model not in the catalog");
    }
}

#[test]
fn should_read_the_same_fields_the_json_carries() {
    let problem_report = ProblemReport {
        app_version: "0.2.126".to_owned(),
        mac: this_test_mac(),
        model: Some(CatalogModelName::from_model_id(
            "Ornith-1.5-35B-A3B-OptiQ-4bit",
            &catalog_ids(),
        )),
        problem: ProblemKind::ModelLoadNotEnoughMemory,
        memory: Some(MemoryShortfall::from_bytes(30_064_771_072, 22_548_578_304)),
    };
    assert_eq!(
        problem_report.readable_lines(),
        [
            "App: Astronomical 0.2.126",
            "Mac: Apple M5 Pro, 48 GB, macOS 27.2",
            "Model: mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit",
            "Problem: model load failed, not enough memory",
            "Memory: needed 28 GB, 21 GB free",
        ]
    );
    let report_json = serde_json::to_value(&problem_report).expect("report should serialize");
    assert_eq!(
        report_json,
        serde_json::json!({
            "app_version": "0.2.126",
            "mac": {"chip_name": "Apple M5 Pro", "memory_gb": 48, "macos_version": "27.2"},
            "model": "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit",
            "problem": "model_load_not_enough_memory",
            "memory": {"needed_gb": 28.0, "available_gb": 21.0}
        })
    );
}

#[test]
fn should_leave_out_the_model_and_memory_lines_when_a_problem_has_neither() {
    let problem_report = ProblemReport {
        app_version: "0.2.126".to_owned(),
        mac: this_test_mac(),
        model: None,
        problem: ProblemKind::EngineDidNotStart,
        memory: None,
    };
    assert_eq!(
        problem_report.readable_lines(),
        [
            "App: Astronomical 0.2.126",
            "Mac: Apple M5 Pro, 48 GB, macOS 27.2",
            "Problem: engine did not start",
        ]
    );
    let report_json = serde_json::to_value(&problem_report).expect("report should serialize");
    assert!(report_json.get("model").is_none());
    assert!(report_json.get("memory").is_none());
}

#[test]
fn should_round_memory_figures_to_one_decimal() {
    let memory_shortfall = MemoryShortfall::from_bytes(1_181_116_006, 0);
    assert_eq!(memory_shortfall.needed_gb, 1.1);
    assert_eq!(memory_shortfall.available_gb, 0.0);
}
