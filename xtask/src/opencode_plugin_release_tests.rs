use crate::opencode_plugin_release::{
    changelog_has_version_heading, parse_release_check_arguments, release_problems,
    version_from_tag,
};

fn arguments(values: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for value in values {
        owned_arguments.push((*value).to_owned());
    }
    owned_arguments
}

const MANIFEST_AT_0_2_0: &str = r#"{ "name": "opencode-astronomical", "version": "0.2.0" }"#;
const CHANGELOG_WITH_0_2_0: &str = "# Changelog\n\n## 0.2.0 (2026-11-19)\n\n- Options.\n";

#[test]
fn should_accept_a_tag_whose_version_matches_the_manifest_and_the_changelog() {
    assert!(
        release_problems(
            "opencode-plugin-v0.2.0",
            MANIFEST_AT_0_2_0,
            CHANGELOG_WITH_0_2_0
        )
        .is_empty()
    );
}

#[test]
fn should_read_the_version_only_from_a_plugin_tag() {
    assert_eq!(version_from_tag("opencode-plugin-v1.10.3"), Some("1.10.3"));
    assert_eq!(version_from_tag("v0.2.118"), None);
    assert_eq!(version_from_tag("opencode-plugin-0.2.0"), None);
}

#[test]
fn should_refuse_a_version_with_a_suffix_missing_part_or_letters() {
    for tag in [
        "opencode-plugin-v0.2.0-beta.1",
        "opencode-plugin-v0.2",
        "opencode-plugin-v0.2.0.1",
        "opencode-plugin-v0.x.0",
        "opencode-plugin-v",
        "opencode-plugin-v0..1",
    ] {
        assert_eq!(version_from_tag(tag), None, "accepted {tag}");
    }
}

#[test]
fn should_refuse_an_app_release_tag_with_one_clear_reason() {
    let problems = release_problems("v0.2.118", MANIFEST_AT_0_2_0, CHANGELOG_WITH_0_2_0);

    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("is not opencode-plugin-vMAJOR.MINOR.PATCH"));
}

#[test]
fn should_report_both_a_manifest_mismatch_and_a_missing_changelog_heading() {
    let problems = release_problems(
        "opencode-plugin-v0.3.0",
        MANIFEST_AT_0_2_0,
        CHANGELOG_WITH_0_2_0,
    );

    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems[0].contains("tag names 0.3.0 but integrations/opencode/package.json says 0.2.0")
    );
    assert!(problems[1].contains("has no \"## 0.3.0\" heading"));
}

#[test]
fn should_report_a_manifest_without_a_string_version() {
    for manifest_text in ["{}", r#"{ "version": 2 }"#, "not json"] {
        let problems = release_problems(
            "opencode-plugin-v0.2.0",
            manifest_text,
            CHANGELOG_WITH_0_2_0,
        );
        assert_eq!(
            problems,
            vec!["integrations/opencode/package.json has no string \"version\"".to_owned()]
        );
    }
}

#[test]
fn should_match_a_changelog_heading_exactly_not_a_longer_version() {
    assert!(changelog_has_version_heading("## 0.2.0\n", "0.2.0"));
    assert!(changelog_has_version_heading(
        "## 0.2.0 (2026-11-19)\n",
        "0.2.0"
    ));
    assert!(!changelog_has_version_heading("## 0.2.01\n", "0.2.0"));
    assert!(!changelog_has_version_heading("## 10.2.0\n", "0.2.0"));
    assert!(!changelog_has_version_heading("### 0.2.0\n", "0.2.0"));
    assert!(!changelog_has_version_heading(
        "- fixed in 0.2.0\n",
        "0.2.0"
    ));
}

#[test]
fn should_read_a_tag_with_or_without_the_main_requirement() {
    let plain_request = parse_release_check_arguments(&arguments(&["opencode-plugin-v0.1.0"]))
        .expect("a lone tag is valid");
    assert_eq!(plain_request.tag, "opencode-plugin-v0.1.0");
    assert!(!plain_request.requires_commit_on_main);

    for flag_order in [
        ["--require-on-main", "opencode-plugin-v0.1.0"],
        ["opencode-plugin-v0.1.0", "--require-on-main"],
    ] {
        let request = parse_release_check_arguments(&arguments(&flag_order))
            .expect("the flag may come before or after the tag");
        assert!(request.requires_commit_on_main);
        assert_eq!(request.tag, "opencode-plugin-v0.1.0");
    }
}

#[test]
fn should_refuse_an_unknown_option_and_a_missing_or_second_tag() {
    assert!(
        parse_release_check_arguments(&arguments(&["--force", "opencode-plugin-v0.1.0"]))
            .unwrap_err()
            .contains("unknown option --force")
    );
    assert!(parse_release_check_arguments(&arguments(&[])).is_err());
    assert!(parse_release_check_arguments(&arguments(&["--require-on-main"])).is_err());
    assert!(
        parse_release_check_arguments(&arguments(&[
            "opencode-plugin-v0.1.0",
            "opencode-plugin-v0.2.0"
        ]))
        .is_err()
    );
}
