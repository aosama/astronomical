//! Acceptance coverage for deterministic prompt-processing chunk configuration.

use super::*;

#[test]
fn should_default_to_fixed_2048_prompt_processing_chunks() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(temp_home.path(), "{}");

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("default config should load")
        .chunking()
        .expect("default chunking should resolve");

    assert_eq!(chunking.fixed_prompt_processing_chunk_size_tokens(), 2_048);
    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        None
    );
}

#[test]
fn should_accept_fixed_resident_and_ssd_streaming_chunk_sizes() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"fixed_prompt_processing_chunk_size_tokens":4096,"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":256}}"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("fixed config should load")
        .chunking()
        .expect("fixed chunking should resolve");

    assert_eq!(chunking.fixed_prompt_processing_chunk_size_tokens(), 4_096);
    assert_eq!(
        chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens(),
        Some(256)
    );
}

#[test]
fn should_reject_zero_fixed_chunk_sizes() {
    for chunking_document in [
        r#"{"fixed_prompt_processing_chunk_size_tokens":0}"#,
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
fn should_reject_an_ssd_streaming_chunk_larger_than_the_resident_chunk() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"chunking":{"fixed_prompt_processing_chunk_size_tokens":2048,"fixed_ssd_streaming_prompt_processing_chunk_size_tokens":4096}}"#,
    );

    assert!(AstronomicalConfig::load_from_home_directory(temp_home.path()).is_err());
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
