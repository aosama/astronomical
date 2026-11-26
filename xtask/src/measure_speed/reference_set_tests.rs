use super::{ReferenceCase, load_reference_set, parse_reference_set};

#[test]
fn should_read_each_case_with_its_memory_limit() {
    let reference_cases = parse_reference_set(
        r#"{"cases": [
            {"model_id": "Model-A", "maximum_mlx_memory_gb": null, "why": "fits in memory"},
            {"model_id": "Model-A", "maximum_mlx_memory_gb": 16, "long_prompt_tokens": 2000, "why": "streams"}
        ]}"#,
    )
    .expect("a valid set");

    assert_eq!(
        reference_cases,
        vec![
            ReferenceCase {
                model_id: "Model-A".to_owned(),
                maximum_mlx_memory_gb: None,
                long_prompt_tokens: None,
                why: "fits in memory".to_owned(),
            },
            ReferenceCase {
                model_id: "Model-A".to_owned(),
                maximum_mlx_memory_gb: Some(16),
                long_prompt_tokens: Some(2_000),
                why: "streams".to_owned(),
            },
        ]
    );
}

#[test]
fn should_refuse_an_empty_set_a_repeated_case_and_unknown_fields() {
    for invalid_text in [
        r#"{"cases": []}"#,
        r#"{"cases": [
            {"model_id": "Model-A", "maximum_mlx_memory_gb": 16, "why": "a"},
            {"model_id": "Model-A", "maximum_mlx_memory_gb": 16, "why": "b"}
        ]}"#,
        r#"{"cases": [{"model_id": "Model-A", "maximum_mlx_memory_gb": 16, "why": "a", "rounds": 9}]}"#,
        r#"{"cases": [{"model_id": "Model-A", "maximum_mlx_memory_gb": 16}]}"#,
    ] {
        assert!(
            parse_reference_set(invalid_text).is_err(),
            "should refuse {invalid_text}"
        );
    }
}

#[test]
fn should_read_the_shipped_reference_set() {
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    let reference_cases = load_reference_set(&repository_root).expect("the shipped set reads");

    assert!(
        reference_cases
            .iter()
            .all(|reference_case| !reference_case.why.is_empty())
    );
}

#[test]
fn should_measure_the_slow_reading_dense_cases_on_a_shorter_long_prompt() {
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let reference_cases = load_reference_set(&repository_root).expect("the shipped set reads");

    for reference_case in &reference_cases {
        let is_the_dense_model = reference_case.model_id.starts_with("Qwen3.8-27B");
        assert_eq!(
            reference_case.long_prompt_tokens,
            is_the_dense_model.then_some(2_000),
            "{}: only the dense 27B reads slowly enough to need a shorter prompt",
            reference_case.model_id
        );
    }
}
