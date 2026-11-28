//! Acceptance coverage for deterministic prompt-processing chunk configuration.

use super::*;

const VERSIONED_HEADER: &str = r#""$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}"#;

fn persisted_config_json(home_directory: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(home_directory.join(".astronomical/config.json"))
            .expect("the config should be readable"),
    )
    .expect("the config should be JSON")
}

fn loaded_in_memory_chunk(home_directory: &std::path::Path) -> Option<u32> {
    AstronomicalConfig::load_from_home_directory(home_directory)
        .expect("the config should load")
        .chunking()
        .expect("chunking should resolve")
        .prompt_processing_chunk_size_tokens()
}

#[test]
fn should_default_to_an_automatic_in_memory_chunk_and_the_platform_ssd_streaming_chunk() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(temp_home.path(), "{}");

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("default config should load")
        .chunking()
        .expect("default chunking should resolve");

    assert_eq!(chunking.prompt_processing_chunk_size_tokens(), None);
    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        config::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS
    );
    let persisted_config = persisted_config_json(temp_home.path());
    assert_eq!(
        persisted_config["chunking"]["prompt_processing_chunk_size_tokens"],
        serde_json::Value::Null,
        "the in-memory chunk should be written as Automatic so config.json shows it"
    );
    assert_eq!(
        persisted_config["chunking"]["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"],
        config::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS
    );
}

#[test]
fn should_turn_a_default_an_earlier_build_wrote_under_the_old_key_into_automatic() {
    for written_default in [2_048, 4_096] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(
                r#"{{{VERSIONED_HEADER},"chunking":{{"fixed_prompt_processing_chunk_size_tokens":{written_default}}}}}"#
            ),
        );

        assert_eq!(loaded_in_memory_chunk(temp_home.path()), None);
        let persisted_chunking = &persisted_config_json(temp_home.path())["chunking"];
        assert_eq!(
            persisted_chunking["prompt_processing_chunk_size_tokens"],
            serde_json::Value::Null
        );
        assert!(
            persisted_chunking
                .get("fixed_prompt_processing_chunk_size_tokens")
                .is_none(),
            "the old key should be gone from the file"
        );
    }
}

#[test]
fn should_move_a_size_no_build_wrote_from_the_old_key_unchanged() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        &format!(
            r#"{{{VERSIONED_HEADER},"chunking":{{"fixed_prompt_processing_chunk_size_tokens":8192}}}}"#
        ),
    );

    assert_eq!(loaded_in_memory_chunk(temp_home.path()), Some(8_192));
    assert_eq!(
        persisted_config_json(temp_home.path())["chunking"]["prompt_processing_chunk_size_tokens"],
        8_192
    );
}

#[test]
fn should_keep_a_2048_the_person_set_under_the_new_key() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        &format!(
            r#"{{{VERSIONED_HEADER},"chunking":{{"prompt_processing_chunk_size_tokens":2048}}}}"#
        ),
    );

    assert_eq!(loaded_in_memory_chunk(temp_home.path()), Some(2_048));
    assert_eq!(
        loaded_in_memory_chunk(temp_home.path()),
        Some(2_048),
        "a second load must not turn the person's 2,048 into Automatic"
    );
}

#[test]
fn should_move_a_model_size_unchanged_even_when_it_equals_an_old_default() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        &format!(
            r#"{{{VERSIONED_HEADER},"chunking":{{"fixed_prompt_processing_chunk_size_tokens":4096}},"models":{{"target":{{"chunking":{{"fixed_prompt_processing_chunk_size_tokens":4096}}}}}}}}"#
        ),
    );

    let astronomical_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("a config with a per-model value should load");

    assert_eq!(
        astronomical_config
            .chunking()
            .expect("global chunking should resolve")
            .prompt_processing_chunk_size_tokens(),
        None
    );
    assert_eq!(
        astronomical_config
            .resolved_model_config("target", 65_536)
            .expect("model policy should resolve")
            .chunking()
            .prompt_processing_chunk_size_tokens(),
        Some(4_096),
        "a per-model value is the person's own and must survive the move"
    );
}

#[test]
fn should_let_a_model_ask_for_automatic_over_a_global_size() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        &format!(
            r#"{{{VERSIONED_HEADER},"chunking":{{"prompt_processing_chunk_size_tokens":6144}},"models":{{"target":{{"chunking":{{"prompt_processing_chunk_size_tokens":null}}}}}}}}"#
        ),
    );

    let model_chunk = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("the config should load")
        .resolved_model_config("target", 65_536)
        .expect("model policy should resolve")
        .chunking()
        .prompt_processing_chunk_size_tokens();

    assert_eq!(model_chunk, None);
}

#[test]
fn should_accept_fixed_resident_and_ssd_streaming_chunk_sizes() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"prompt_processing_chunk_size_tokens":6144,"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":256}}"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("fixed config should load")
        .chunking()
        .expect("fixed chunking should resolve");

    assert_eq!(chunking.prompt_processing_chunk_size_tokens(), Some(6_144));
    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        256
    );
}

#[test]
fn should_reject_zero_fixed_chunk_sizes() {
    for chunking_document in [
        r#"{"prompt_processing_chunk_size_tokens":0}"#,
        r#"{"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":0}"#,
    ] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(r#"{{"chunking":{chunking_document}}}"#),
        );

        assert!(AstronomicalConfig::load_from_home_directory(temp_home.path()).is_err());
    }
}

#[test]
fn should_accept_an_ssd_streaming_chunk_larger_than_the_resident_chunk() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"prompt_processing_chunk_size_tokens":1024,"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":4096}}"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("a larger paged chunk should load")
        .chunking()
        .expect("chunking should resolve");
    assert_eq!(chunking.prompt_processing_chunk_size_tokens(), Some(1_024));
    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        4_096
    );
}

#[test]
fn should_reject_every_retired_optimizer_configuration_key() {
    for retired_key in [
        "prompt_processing_chunk_size_optimizer_enabled",
        "prompt_processing_chunk_size_optimizer_candidate_token_counts",
        "prompt_processing_chunk_size_optimizer_maximum_retained_measurements_per_candidate_and_context",
        "prompt_processing_chunk_size_optimizer_position_range_size_tokens",
    ] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(r#"{{"chunking":{{"{retired_key}":true}}}}"#),
        );

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temp_home.path()),
            Err(AstronomicalConfigError::ParseConfigFile { .. })
        ));
    }
}

#[test]
fn should_carry_the_retired_ssd_streaming_chunk_default_forward() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":2048}}"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("a config with the retired default should load")
        .chunking()
        .expect("chunking should resolve");

    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        config::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS
    );
}

#[test]
fn should_keep_a_deliberate_ssd_streaming_chunk_value() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":1024}}"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("a config with a deliberate chunk should load")
        .chunking()
        .expect("chunking should resolve");

    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        1_024
    );
}

#[test]
fn should_materialize_the_ssd_streaming_chunk_only_when_absent_or_retired() {
    use config::should_materialize_ssd_streaming_chunk_default;
    assert!(should_materialize_ssd_streaming_chunk_default(None));
    assert!(should_materialize_ssd_streaming_chunk_default(Some(2_048)));
    assert!(!should_materialize_ssd_streaming_chunk_default(Some(4_096)));
    assert!(!should_materialize_ssd_streaming_chunk_default(Some(2_047)));
    assert!(!should_materialize_ssd_streaming_chunk_default(Some(2_049)));
}
