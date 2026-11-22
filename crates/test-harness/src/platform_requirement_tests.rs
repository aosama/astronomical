use super::{PlatformRequirement, skip_without};

#[test]
fn should_run_tests_on_macos_and_skip_them_elsewhere() {
    for platform_requirement in [
        PlatformRequirement::CoreMl,
        PlatformRequirement::MetalGpu,
        PlatformRequirement::MacosSystemTools,
        PlatformRequirement::MacosMenuApp,
    ] {
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
    assert!(
        PlatformRequirement::MacosSystemTools
            .description()
            .contains("macOS")
    );
}
