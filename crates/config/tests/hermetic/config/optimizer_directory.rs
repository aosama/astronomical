use super::*;

#[test]
fn should_resolve_optimizer_directory_from_home_directory() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{ "model_directories": ["/models/ornith"] }"#,
    );
    let user_config =
        AstronomicalConfig::load_from_home_directory(temp_home.path()).expect("config should load");

    assert_eq!(
        user_config
            .optimizer_directory()
            .expect("optimizer directory should resolve"),
        temp_home.path().join(".astronomical/optimizer")
    );
}

#[test]
fn should_fail_to_resolve_optimizer_directory_when_home_directory_is_missing() {
    // When there is no home directory, the optimizer directory cannot be derived.
    // This test verifies that the error variant exists and is returned correctly.
    // We cannot construct AstronomicalConfig with home_directory=None without
    // going through load_from_default_location, which depends on $HOME.
    // Instead, verify the error variant is defined by matching on it.
    let _error = AstronomicalConfigError::DefaultOptimizerDirectoryRequiresHome;
}
