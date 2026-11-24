use super::*;

#[test]
fn should_enable_the_standard_astro_ssd_prompt_cache_with_the_default_capacity() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(temporary_home_directory.path(), r#"{}"#);

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .astro_ssd_prompt_cache()
            .expect("astro SSD prompt cache should resolve"),
        AstroSsdPromptCacheConfig::new(
            temporary_home_directory.path().join(".astronomical/cache"),
            53_687_091_200,
        )
    );
    assert!(astronomical_config.astro_ssd_prompt_cache_enabled());
}

/// A file written before the astro SSD prompt cache got its formal name holds
/// `prompt_cache` keys, and refusing it would stop the daemon on update.
#[test]
fn should_load_a_config_holding_the_old_prompt_cache_keys_and_rewrite_it_under_the_new_ones() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "prompt_cache": {"enabled": false, "maximum_size_gb": 20},
          "chunking": {"prompt_cache_common_prefix_stride_blocks": 8},
          "models": {"target": {"chunking": {"prompt_cache_block_tokens": 256}}}
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the old prompt_cache keys must never stop a config from loading");

    assert!(!astronomical_config.astro_ssd_prompt_cache_enabled());
    let model_config = astronomical_config
        .resolved_model_config("target", 65_536)
        .expect("the model policy should resolve");
    assert_eq!(
        model_config.chunking().ssd_prompt_cache_block_tokens(),
        Some(256)
    );
    assert_eq!(
        model_config
            .chunking()
            .ssd_prompt_cache_common_prefix_stride_blocks(),
        8
    );
    let rewritten_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            temporary_home_directory
                .path()
                .join(".astronomical/config.json"),
        )
        .expect("the rewritten config should be readable"),
    )
    .expect("the rewritten config should be JSON");
    assert_eq!(rewritten_config["ssd_prompt_cache"]["maximum_size_gb"], 20);
    assert!(rewritten_config.get("prompt_cache").is_none());
    assert_eq!(
        rewritten_config["chunking"]["ssd_prompt_cache_common_prefix_stride_blocks"],
        8
    );
    assert_eq!(
        rewritten_config["models"]["target"]["chunking"]["ssd_prompt_cache_block_tokens"],
        256
    );
}

#[test]
fn should_refuse_a_config_whose_prompt_cache_section_holds_an_unknown_setting() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "prompt_cache": {"enabled": true, "not_a_setting": 1}
        }"#,
    );

    let load_error = AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect_err("a moved section is still checked as strictly as any other");

    assert!(
        load_error.to_string().contains("not_a_setting"),
        "the error should name the unknown setting: {load_error}"
    );
}
