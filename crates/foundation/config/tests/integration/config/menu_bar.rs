use super::*;

fn config_with_menu_bar(menu_bar_section: &str) -> String {
    format!(
        r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[]}},"menu_bar":{menu_bar_section}}}"#
    )
}

#[test]
fn should_show_only_the_menu_bar_icon_when_the_setting_is_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should load");

    assert!(!astronomical_config.menu_bar_shows_progress_and_speed());
    assert!(!astronomical_config.menu_bar_shows_technical_details());
}

#[test]
fn should_read_each_authored_menu_bar_switch_and_an_empty_section() {
    for (menu_bar_section, expected_setting) in [
        (r#"{"show_progress_and_speed":true}"#, (true, false)),
        (r#"{"show_technical_details":true}"#, (false, true)),
        (
            r#"{"show_progress_and_speed":true,"show_technical_details":true}"#,
            (true, true),
        ),
        (r#"{"show_progress_and_speed":false}"#, (false, false)),
        ("{}", (false, false)),
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &config_with_menu_bar(menu_bar_section),
        );

        let astronomical_config =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
                .expect("config should load");

        assert_eq!(
            (
                astronomical_config.menu_bar_shows_progress_and_speed(),
                astronomical_config.menu_bar_shows_technical_details(),
            ),
            expected_setting,
            "{menu_bar_section}"
        );
    }
}

#[test]
fn should_refuse_a_menu_bar_setting_that_is_not_a_boolean_or_is_unknown() {
    for menu_bar_section in [
        r#"{"show_progress_and_speed":"yes"}"#,
        r#"{"show_progress_and_speed":1}"#,
        r#"{"show_technical_details":"on"}"#,
        r#"{"show_speed":true}"#,
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &config_with_menu_bar(menu_bar_section),
        );

        let loading_outcome =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path());

        assert!(
            loading_outcome.is_err(),
            "{menu_bar_section} should be refused"
        );
    }
}

#[test]
fn should_match_the_schema_defaults_and_list_both_switches_on_the_settings_page() {
    let schema_document: serde_json::Value = serde_json::from_slice(config::config_schema_bytes())
        .expect("the embedded schema should be JSON");
    let settings_page_text = schema_document["x-astronomical-layout"]["settings"].to_string();
    for switch_name in ["show_progress_and_speed", "show_technical_details"] {
        let schema_property = &schema_document["$defs"]["menu_bar"]["properties"][switch_name];
        assert_eq!(schema_property["default"], false, "{switch_name}");
        assert_eq!(
            schema_property["x-astronomical-apply-mode"], "live",
            "{switch_name}"
        );
        assert!(settings_page_text.contains(&format!("menu_bar.{switch_name}")));
    }
}

fn configuration_generation_for(document: &str) -> String {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(temporary_home_directory.path(), document);
    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("config should load")
        .generation()
        .to_owned()
}

#[test]
fn should_keep_the_configuration_generation_whatever_the_menu_bar_section_holds() {
    // A new generation that no reload field explains replaces the worker and unloads the
    // model; the menu bar switches must never cause one.
    let without_section = configuration_generation_for(
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}}"#,
    );
    for menu_bar_section in [
        "{}",
        r#"{"show_progress_and_speed":false}"#,
        r#"{"show_progress_and_speed":true,"show_technical_details":true}"#,
    ] {
        assert_eq!(
            configuration_generation_for(&config_with_menu_bar(menu_bar_section)),
            without_section,
            "{menu_bar_section}"
        );
    }
}

#[test]
fn should_still_change_the_configuration_generation_for_a_worker_setting_beside_the_menu_bar() {
    let menu_bar_only =
        configuration_generation_for(&config_with_menu_bar(r#"{"show_technical_details":true}"#));
    let with_astro_ssd_prompt_cache_off = configuration_generation_for(
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"ssd_prompt_cache":{"enabled":false},"menu_bar":{"show_technical_details":true}}"#,
    );
    assert_ne!(menu_bar_only, with_astro_ssd_prompt_cache_off);
}
