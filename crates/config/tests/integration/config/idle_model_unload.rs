use super::*;

fn config_with_idle_unload(authored_value: &str) -> String {
    format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[],"idle_model_unload_seconds":{authored_value}}}}}"#
    )
}

#[test]
fn should_default_the_idle_unload_span_to_sixty_seconds_when_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");

    assert_eq!(astronomical_config.idle_model_unload_seconds(), 60);
}

#[test]
fn should_read_an_authored_idle_unload_span_including_zero() {
    for (authored_value, expected_seconds) in [("300", 300), ("0", 0), ("604800", 604_800)] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &config_with_idle_unload(authored_value),
        );

        let astronomical_config =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
                .expect("config should load");

        assert_eq!(
            astronomical_config.idle_model_unload_seconds(),
            expected_seconds
        );
    }
}

#[test]
fn should_refuse_an_idle_unload_span_that_is_negative_fractional_text_or_above_one_week() {
    for authored_value in ["-1", "\"sixty\"", "1.5", "604801"] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &config_with_idle_unload(authored_value),
        );

        let loading_outcome =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path());

        assert!(
            loading_outcome.is_err(),
            "{authored_value} should be refused"
        );
    }
}

#[test]
fn should_match_the_schema_default_and_maximum_for_the_idle_unload_span() {
    let schema_document: serde_json::Value =
        serde_json::from_slice(astronomical_config::config_schema_bytes())
            .expect("the embedded schema should be JSON");
    let schema_property =
        &schema_document["$defs"]["runtime"]["properties"]["idle_model_unload_seconds"];

    assert_eq!(
        schema_property["default"],
        astronomical_config::DEFAULT_IDLE_MODEL_UNLOAD_SECONDS
    );
    assert_eq!(
        schema_property["maximum"],
        astronomical_config::MAXIMUM_IDLE_MODEL_UNLOAD_SECONDS
    );
}
