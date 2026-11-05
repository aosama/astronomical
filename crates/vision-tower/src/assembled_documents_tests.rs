use serde_json::json;

use super::*;

fn with_facts<Answer>(
    tower_config: &Value,
    tensor_names: &[String],
    use_facts: fn(&TowerDocumentFacts<'_>) -> Answer,
) -> Answer {
    use_facts(&TowerDocumentFacts {
        tower_config,
        tensor_names,
        tensor_byte_count: 100,
        parameter_count: 50,
    })
}

fn tower_names() -> Vec<String> {
    vec!["vision_tower.a".to_owned(), "vision_tower.b".to_owned()]
}

fn merged_with_tokens(facts: &TowerDocumentFacts<'_>) -> Value {
    config_with_tower(
        &json!({"hidden_size": 8, "image_token_id": 1, "quantization": {"bits": 4}}),
        facts,
    )
}

#[test]
fn should_add_the_vision_section_and_leave_text_settings_alone() {
    let tower_config =
        json!({"vision_config": {"depth": 27}, "image_token_id": 99, "video_token_id": 2});
    let merged = with_facts(&tower_config, &tower_names(), merged_with_tokens);
    assert_eq!(merged["vision_config"]["depth"], 27);
    assert_eq!(merged["hidden_size"], 8);
    assert_eq!(merged["quantization"]["bits"], 4);
    assert_eq!(
        merged["image_token_id"], 1,
        "the text model's own value stands"
    );
    assert_eq!(
        merged["video_token_id"], 2,
        "a missing token comes from the tower"
    );
    assert!(
        merged.get("optiq_vision").is_none(),
        "this is not an OptiQ build"
    );
}

fn index_from_existing(facts: &TowerDocumentFacts<'_>) -> Value {
    index_with_tower(
        Some(&json!({
            "metadata": {"total_size": 1000, "total_parameters": 500},
            "weight_map": {"text.weight": "model.safetensors"},
        })),
        &[],
        facts,
    )
}

#[test]
fn should_extend_an_existing_index_and_its_totals() {
    let index = with_facts(&json!({}), &tower_names(), index_from_existing);
    assert_eq!(index["weight_map"]["text.weight"], "model.safetensors");
    assert_eq!(
        index["weight_map"]["vision_tower.a"],
        TOWER_FILE_RELATIVE_PATH
    );
    assert_eq!(index["metadata"]["total_size"], 1100);
    assert_eq!(index["metadata"]["total_parameters"], 550);
}

fn index_from_nothing(facts: &TowerDocumentFacts<'_>) -> Value {
    index_with_tower(
        None,
        &[("text.weight".to_owned(), "model.safetensors".to_owned())],
        facts,
    )
}

#[test]
fn should_build_an_index_when_the_text_model_has_none() {
    let index = with_facts(&json!({}), &tower_names(), index_from_nothing);
    assert_eq!(index["weight_map"]["text.weight"], "model.safetensors");
    assert_eq!(
        index["weight_map"]["vision_tower.b"],
        TOWER_FILE_RELATIVE_PATH
    );
    assert_eq!(index["metadata"]["total_size"], 100);
    assert!(index["metadata"].get("total_parameters").is_none());
}

#[test]
fn should_name_the_tower_file_as_published_models_do() {
    assert!(
        !TOWER_FILE_RELATIVE_PATH.contains('/'),
        "a downloader may flatten folders"
    );
    assert!(TOWER_FILE_RELATIVE_PATH.starts_with("model"));
    assert!(TOWER_FILE_RELATIVE_PATH.ends_with(".safetensors"));
    assert!(!TOWER_FILE_RELATIVE_PATH.contains("optiq"));
}
