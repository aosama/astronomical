use super::*;

astronomical_test_harness::offline_test!(
    should_enable_the_standard_prompt_cache_with_the_default_capacity,
    {
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
                53_687_091_200,
            )
        );
        assert!(astronomical_config.persistent_prompt_cache_enabled());
    }
);
