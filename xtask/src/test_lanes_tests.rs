use std::collections::BTreeMap;

use super::{LaneDeclaration, lane_alias_text, lane_declarations_from_metadata};

fn metadata_for(packages: &[(&str, &str)]) -> String {
    let package_documents: Vec<String> = packages
        .iter()
        .map(|(package_name, metadata_json)| {
            format!(r#"{{"name": "{package_name}", "metadata": {metadata_json}}}"#)
        })
        .collect();
    format!(r#"{{"packages": [{}]}}"#, package_documents.join(","))
}

fn declarations(entries: &[(&str, &[&str], Option<&str>)]) -> BTreeMap<String, LaneDeclaration> {
    entries
        .iter()
        .map(|(package_name, lanes, feature)| {
            (
                (*package_name).to_owned(),
                LaneDeclaration {
                    test_lanes: lanes.iter().map(|lane| (*lane).to_owned()).collect(),
                    direct_mlx_lane_feature: feature.map(str::to_owned),
                },
            )
        })
        .collect()
}

#[test]
fn should_read_each_package_lanes_and_lane_feature() {
    let lane_declarations = lane_declarations_from_metadata(&metadata_for(&[
        (
            "engine",
            r#"{"astronomical": {"test-lanes": ["integration", "direct-mlx"], "direct-mlx-lane-feature": "mlx"}}"#,
        ),
        ("tooling", r#"{"astronomical": {"test-lanes": []}}"#),
    ]))
    .expect("declared packages should read");

    assert_eq!(
        lane_declarations,
        declarations(&[
            ("engine", &["integration", "direct-mlx"], Some("mlx")),
            ("tooling", &[], None),
        ])
    );
}

#[test]
fn should_refuse_a_package_that_declares_no_lanes() {
    let refusal = lane_declarations_from_metadata(&metadata_for(&[
        (
            "engine",
            r#"{"astronomical": {"test-lanes": ["integration"]}}"#,
        ),
        ("forgotten", "null"),
    ]))
    .expect_err("an undeclared package must fail the check");

    assert!(refusal.contains("forgotten"), "{refusal}");
}

#[test]
fn should_refuse_an_unknown_lane_name() {
    let refusal = lane_declarations_from_metadata(&metadata_for(&[(
        "engine",
        r#"{"astronomical": {"test-lanes": ["integratoin"]}}"#,
    )]))
    .expect_err("a misspelt lane must fail the check");

    assert!(refusal.contains("integratoin"), "{refusal}");
}

#[test]
fn should_build_a_lane_alias_from_its_packages() {
    let lane_declarations = declarations(&[
        ("core", &["integration"], None),
        ("engine", &["integration", "direct-mlx"], Some("direct-mlx")),
        ("tooling", &[], None),
    ]);

    assert_eq!(
        lane_alias_text("test-integration", &lane_declarations),
        Some(
            "test --no-fail-fast -p core -p engine --lib --test integration_tests --test hardware_journeys"
                .to_owned()
        )
    );
}

#[test]
fn should_add_the_direct_mlx_feature_only_where_a_package_names_one() {
    let lane_declarations = declarations(&[
        ("engine", &["direct-mlx"], Some("direct-mlx")),
        ("probe", &["direct-mlx"], None),
    ]);

    assert_eq!(
        lane_alias_text("test-direct-mlx", &lane_declarations),
        Some(
            "test --no-fail-fast -p engine --features engine/direct-mlx -p probe --lib --test direct_mlx --test mlx_c_coverage_contract_tests -- --test-threads=1"
                .to_owned()
        )
    );
}

#[test]
fn should_not_generate_an_alias_no_lane_defines() {
    assert_eq!(lane_alias_text("test-rest-api", &BTreeMap::new()), None);
}
