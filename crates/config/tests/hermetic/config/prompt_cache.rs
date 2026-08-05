use super::*;

#[test]
fn should_enable_the_standard_prompt_cache_with_the_default_capacity() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(temporary_home_directory.path(), r#"{}"#);

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .prompt_cache()
            .expect("prompt cache should resolve"),
        PromptCacheConfig::new(
            temporary_home_directory.path().join(".astronomical/cache"),
            50_000_000_000,
        )
    );
    assert!(astronomical_config.persistent_prompt_cache_enabled());
}

#[test]
fn should_disable_the_persistent_prompt_cache_when_explicitly_configured() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"persistent_prompt_cache_enabled": false}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.persistent_prompt_cache_enabled());
}

#[test]
fn should_use_the_configured_prompt_cache_capacity_and_standard_root() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"prompt_cache_max_size_gb": 20}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .prompt_cache()
            .expect("prompt cache should resolve"),
        PromptCacheConfig::new(
            temporary_home_directory.path().join(".astronomical/cache"),
            20_000_000_000,
        )
    );
}
