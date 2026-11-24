use super::{PlatformRequirement, skip_without};

#[test]
fn should_run_tests_on_macos_and_skip_them_elsewhere() {
    for platform_requirement in every_requirement() {
        assert_eq!(
            skip_without(platform_requirement, "a requirement test"),
            !cfg!(target_os = "macos"),
            "{platform_requirement:?} decides by the target platform"
        );
    }
}

#[test]
fn should_name_what_is_missing_in_each_skip_reason() {
    assert!(
        PlatformRequirement::CoreMl
            .description()
            .contains("Core ML")
    );
    // Each reason sits at its requirement's discriminant, so a reordered
    // table names the wrong thing; these pin every row to its requirement.
    assert!(
        PlatformRequirement::MetalGpu
            .description()
            .contains("Metal kernel")
    );
    assert!(
        PlatformRequirement::MacosSystemTools
            .description()
            .contains("system tools")
    );
    assert!(
        PlatformRequirement::MlxGpuBackend
            .description()
            .contains("GPU backend")
    );
}

/// Every requirement, checked by an exhaustive match so a new variant fails
/// to compile here until it is listed (and its skip reason written).
fn every_requirement() -> [PlatformRequirement; 4] {
    match PlatformRequirement::CoreMl {
        PlatformRequirement::CoreMl
        | PlatformRequirement::MetalGpu
        | PlatformRequirement::MacosSystemTools
        | PlatformRequirement::MlxGpuBackend => {}
    }
    [
        PlatformRequirement::CoreMl,
        PlatformRequirement::MetalGpu,
        PlatformRequirement::MacosSystemTools,
        PlatformRequirement::MlxGpuBackend,
    ]
}

#[test]
fn should_give_every_requirement_its_own_skip_reason() {
    let every_requirement = every_requirement();
    let distinct_reasons: std::collections::BTreeSet<&str> = every_requirement
        .into_iter()
        .map(PlatformRequirement::description)
        .collect();
    // Boundary: one reason per requirement, none shared and none empty.
    assert_eq!(distinct_reasons.len(), every_requirement.len());
    assert!(!distinct_reasons.contains(""));
}
