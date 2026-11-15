use serde_json::json;

use super::rename_renamed_config_fields;

#[test]
fn should_move_the_prompt_astro_ssd_prompt_cache_section_and_chunking_keys_to_their_new_names() {
    let mut config_json = json!({
        "prompt_cache": {"enabled": true, "maximum_size_gb": 50},
        "chunking": {
            "prompt_cache_block_tokens": null,
            "prompt_cache_common_prefix_stride_blocks": 4,
            "fixed_prompt_processing_chunk_size_tokens": 2048
        },
        "models": {"some-model": {"chunking": {"prompt_cache_block_tokens": 256}}}
    });

    assert!(rename_renamed_config_fields(&mut config_json));

    assert_eq!(
        config_json,
        json!({
            "ssd_prompt_cache": {"enabled": true, "maximum_size_gb": 50},
            "chunking": {
                "ssd_prompt_cache_block_tokens": null,
                "ssd_prompt_cache_common_prefix_stride_blocks": 4,
                "fixed_prompt_processing_chunk_size_tokens": 2048
            },
            "models": {"some-model": {"chunking": {"ssd_prompt_cache_block_tokens": 256}}}
        })
    );
}

#[test]
fn should_keep_the_new_value_when_a_file_holds_both_keys() {
    let mut config_json = json!({
        "prompt_cache": {"maximum_size_gb": 10},
        "ssd_prompt_cache": {"maximum_size_gb": 80}
    });

    assert!(rename_renamed_config_fields(&mut config_json));

    assert_eq!(
        config_json,
        json!({"ssd_prompt_cache": {"maximum_size_gb": 80}})
    );
}

#[test]
fn should_report_nothing_moved_for_a_file_already_on_the_new_keys() {
    let mut config_json = json!({
        "ssd_prompt_cache": {"enabled": false},
        "chunking": {"ssd_prompt_cache_common_prefix_stride_blocks": 4},
        "models": {"some-model": {}}
    });
    let unchanged_config_json = config_json.clone();

    assert!(!rename_renamed_config_fields(&mut config_json));

    assert_eq!(config_json, unchanged_config_json);
}

#[test]
fn should_leave_a_document_that_is_not_an_object_alone() {
    let mut config_json = json!(["prompt_cache"]);

    assert!(!rename_renamed_config_fields(&mut config_json));

    assert_eq!(config_json, json!(["prompt_cache"]));
}
