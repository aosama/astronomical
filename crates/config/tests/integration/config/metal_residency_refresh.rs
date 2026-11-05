use super::*;

fn config_with_refresh(authored_value: &str) -> String {
    format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[],"metal_residency_refresh_enabled":{authored_value}}}}}"#
    )
}

#[test]
fn should_leave_metal_residency_refresh_off_when_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");

    assert!(!astronomical_config.metal_residency_refresh_enabled());
}

#[test]
fn should_read_metal_residency_refresh_when_authored_on() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        &config_with_refresh("true"),
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(astronomical_config.metal_residency_refresh_enabled());
}

#[test]
fn should_read_metal_residency_refresh_when_authored_off() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        &config_with_refresh("false"),
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.metal_residency_refresh_enabled());
}

#[test]
fn should_refuse_a_metal_residency_refresh_that_is_not_a_boolean() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        &config_with_refresh("1000"),
    );

    let loading_outcome =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path());

    assert!(loading_outcome.is_err());
}
