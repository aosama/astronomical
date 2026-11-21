use super::*;

fn config_with_skill(skill_section: &str) -> String {
    format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[]}},"skill":{skill_section}}}"#
    )
}

fn load_config(document: &str) -> Result<AstronomicalConfig, config::AstronomicalConfigError> {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(temporary_home_directory.path(), document);
    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
}

/// Both switches start on: the skill helps a person only when their agent
/// learns of it without any setup.
#[test]
fn should_declare_and_update_the_skill_when_the_setting_is_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");
    assert!(astronomical_config.skill_declared_to_models());
    assert!(astronomical_config.skill_updates_from_web());
}

#[test]
fn should_read_each_authored_skill_switch_and_an_empty_section() {
    for (skill_section, expected_setting) in [
        (r#"{"declare_to_models":false}"#, (false, true)),
        (r#"{"update_from_web":false}"#, (true, false)),
        (
            r#"{"declare_to_models":false,"update_from_web":false}"#,
            (false, false),
        ),
        (r#"{"declare_to_models":true}"#, (true, true)),
        ("{}", (true, true)),
    ] {
        let astronomical_config =
            load_config(&config_with_skill(skill_section)).expect("config should load");
        assert_eq!(
            (
                astronomical_config.skill_declared_to_models(),
                astronomical_config.skill_updates_from_web(),
            ),
            expected_setting,
            "{skill_section}"
        );
    }
}

#[test]
fn should_refuse_a_skill_setting_that_is_not_a_boolean_or_is_unknown() {
    for skill_section in [
        r#"{"declare_to_models":"yes"}"#,
        r#"{"update_from_web":0}"#,
        r#"{"inject":true}"#,
    ] {
        assert!(
            load_config(&config_with_skill(skill_section)).is_err(),
            "{skill_section} should be refused"
        );
    }
}

/// Switching the declaration must never replace the worker and unload the
/// person's model, so the section stays out of the configuration generation.
#[test]
fn should_leave_the_configuration_generation_unchanged_by_the_skill_section() {
    let without_section = load_config(&config_with_skill("{}")).expect("config should load");
    let with_section = load_config(&config_with_skill(
        r#"{"declare_to_models":false,"update_from_web":false}"#,
    ))
    .expect("config should load");
    assert_eq!(without_section.generation(), with_section.generation());
}

#[test]
fn should_match_the_schema_defaults_and_list_both_switches_on_the_settings_page() {
    let schema_document: serde_json::Value = serde_json::from_slice(config::config_schema_bytes())
        .expect("the embedded schema should be JSON");
    let settings_page_text = schema_document["x-astronomical-layout"]["settings"].to_string();
    for switch_name in ["declare_to_models", "update_from_web"] {
        let schema_property = &schema_document["$defs"]["skill"]["properties"][switch_name];
        assert_eq!(schema_property["default"], true, "{switch_name}");
        assert_eq!(
            schema_property["x-astronomical-apply-mode"], "live",
            "{switch_name}"
        );
        assert!(settings_page_text.contains(&format!("skill.{switch_name}")));
    }
}
