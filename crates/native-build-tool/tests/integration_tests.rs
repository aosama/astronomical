//! The native-build-tool's integration-lane entry: the workflow lane
//! contract requires every package in the integration selection to expose
//! this target, so this canary keeps the lane alive while the exhaustive
//! argument-parsing coverage (7 tests) lives in the lib unit tests beside
//! `parse_arguments`.

use astronomical_native_build_tool::parse_arguments;

#[test]
fn should_run_the_native_build_tool_integration_smoke_contract() {
    // One representative parse through the public API, enough to prove the
    // crate's test surface is alive inside the integration lane; the
    // exhaustive coverage lives in the lib unit tests beside
    // `parse_arguments`.
    let parsed_arguments = parse_arguments(&[
        "--profile".to_owned(),
        "core".to_owned(),
        "--repository-root".to_owned(),
        "/".to_owned(),
    ])
    .expect("the core profile smoke arguments should parse");
    assert_eq!(
        parsed_arguments.native_build_profile().identity_name(),
        "core"
    );
}
