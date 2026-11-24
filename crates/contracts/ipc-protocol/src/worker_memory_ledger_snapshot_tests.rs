use super::ActiveMemoryCeilingEnforcement;

/// The two strings `/v1/status` carries and the Linux guide quotes.
#[test]
fn should_name_each_enforcement_on_the_wire_as_the_docs_do() {
    assert_eq!(
        serde_json::to_value(ActiveMemoryCeilingEnforcement::EnforcedByMlx).unwrap(),
        "enforced_by_mlx"
    );
    assert_eq!(
        serde_json::to_value(ActiveMemoryCeilingEnforcement::Unavailable).unwrap(),
        "unavailable"
    );
}

#[test]
fn should_refuse_an_enforcement_name_it_does_not_know() {
    assert!(
        serde_json::from_value::<ActiveMemoryCeilingEnforcement>(serde_json::json!("enforced"))
            .is_err()
    );
    assert!(
        serde_json::from_value::<ActiveMemoryCeilingEnforcement>(serde_json::json!("")).is_err()
    );
}

#[test]
fn should_report_the_patched_allocator_only_on_the_metal_build() {
    let expected_enforcement = if cfg!(target_os = "macos") {
        ActiveMemoryCeilingEnforcement::EnforcedByMlx
    } else {
        ActiveMemoryCeilingEnforcement::Unavailable
    };
    assert_eq!(
        ActiveMemoryCeilingEnforcement::for_this_build(),
        expected_enforcement
    );
}
