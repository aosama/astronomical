use serde_json::json;

use super::move_in_memory_chunk_setting;

fn moved(mut config_json: serde_json::Value) -> (serde_json::Value, Vec<String>) {
    let moved_setting_paths =
        move_in_memory_chunk_setting(config_json.as_object_mut().expect("an object"));
    (config_json, moved_setting_paths)
}

#[test]
fn should_turn_the_default_an_earlier_build_wrote_into_automatic() {
    let (config_json, moved_setting_paths) =
        moved(json!({"chunking": {"fixed_prompt_processing_chunk_size_tokens": 2048}}));

    assert_eq!(
        config_json,
        json!({"chunking": {"prompt_processing_chunk_size_tokens": null}})
    );
    assert_eq!(
        moved_setting_paths,
        ["chunking.fixed_prompt_processing_chunk_size_tokens"]
    );
}

#[test]
fn should_turn_the_retired_4096_default_into_automatic() {
    let (config_json, _) =
        moved(json!({"chunking": {"fixed_prompt_processing_chunk_size_tokens": 4096}}));

    assert_eq!(
        config_json,
        json!({"chunking": {"prompt_processing_chunk_size_tokens": null}})
    );
}

#[test]
fn should_keep_a_global_size_no_build_ever_wrote() {
    let (config_json, _) =
        moved(json!({"chunking": {"fixed_prompt_processing_chunk_size_tokens": 1024}}));

    assert_eq!(
        config_json,
        json!({"chunking": {"prompt_processing_chunk_size_tokens": 1024}})
    );
}

#[test]
fn should_keep_a_model_size_even_when_it_equals_an_old_default() {
    let (config_json, moved_setting_paths) = moved(json!({
        "models": {"some-model": {"chunking": {"fixed_prompt_processing_chunk_size_tokens": 2048}}}
    }));

    assert_eq!(
        config_json,
        json!({"models": {"some-model": {"chunking": {"prompt_processing_chunk_size_tokens": 2048}}}})
    );
    assert_eq!(
        moved_setting_paths,
        ["models.some-model.chunking.fixed_prompt_processing_chunk_size_tokens"]
    );
}

#[test]
fn should_keep_the_new_key_when_a_file_holds_both() {
    let (config_json, _) = moved(json!({"chunking": {
        "fixed_prompt_processing_chunk_size_tokens": 2048,
        "prompt_processing_chunk_size_tokens": 8192
    }}));

    assert_eq!(
        config_json,
        json!({"chunking": {"prompt_processing_chunk_size_tokens": 8192}})
    );
}

#[test]
fn should_move_nothing_in_a_file_already_on_the_new_key() {
    let original = json!({
        "chunking": {"prompt_processing_chunk_size_tokens": 2048},
        "models": {"some-model": {}, "other-model": {"chunking": {}}}
    });

    let (config_json, moved_setting_paths) = moved(original.clone());

    assert_eq!(config_json, original);
    assert!(moved_setting_paths.is_empty());
}

#[test]
fn should_move_a_value_that_is_not_a_number_unchanged_for_the_parser_to_refuse() {
    let (config_json, _) =
        moved(json!({"chunking": {"fixed_prompt_processing_chunk_size_tokens": "big"}}));

    assert_eq!(
        config_json,
        json!({"chunking": {"prompt_processing_chunk_size_tokens": "big"}})
    );
}
