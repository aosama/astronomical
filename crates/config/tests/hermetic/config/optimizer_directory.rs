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
