use super::*;

#[test]
fn should_load_trace_logging_with_bounded_retention() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "logging": {
            "level": "trace",
            "retained_files": 3
          }
        }"#,
    );
    let user_config =
        AstronomicalConfig::load_from_home_directory(temp_home.path()).expect("config should load");

    assert_eq!(
        user_config.logging().expect("logging should resolve"),
        LoggingConfig::new(
            temp_home.path().join(".astronomical/logs"),
            LogLevel::Trace,
            3,
        )
    );
}

#[test]
fn should_reject_zero_retained_log_files() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{"logging":{"level":"debug","retained_files":0}}"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::InvalidRetainedLogFileCount)
    ));
}
