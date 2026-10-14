use astronomical_test_harness::offline_test;
use std::path::Path;

use crate::digest_bytes;
use crate::probes::{ProbeValue, compatibility_manifest_lines, compatibility_probe_values};
use crate::{FingerprintRequest, IdentityScope, NativeBuildProfile};

fn overridden_request() -> FingerprintRequest<'static> {
    FingerprintRequest {
        // Every probe is overridden, so the values are offline-deterministic
        // and no real toolchain is invoked.
        repository_root: Path::new("/"),
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: crate::probes::CompatibilityOverrides {
            xcode: Some("Xcode 26.0 Build 17A1".to_owned()),
            sdk: Some("version=15.4\nbuild=24A123".to_owned()),
            clang: Some("version=17.0.0\ntarget=arm64-apple-darwin24".to_owned()),
            cmake: Some("cmake version 4.2.1".to_owned()),
            rustc: Some("rustc 1.97.1".to_owned()),
            target_triple: Some("aarch64-apple-darwin".to_owned()),
            build_type: Some("Release".to_owned()),
        },
    }
}

offline_test!(probe_values_carry_the_fixed_order_and_override_flag, {
    let probe_values = compatibility_probe_values(&overridden_request()).unwrap();
    let names: Vec<&str> = probe_values.iter().map(|entry| entry.name).collect();
    assert_eq!(
        names,
        [
            "target",
            "build-type",
            "xcode",
            "sdk",
            "clang",
            "cmake",
            "rustc"
        ]
    );
    assert!(
        probe_values.iter().all(|entry| entry.from_override),
        "explicitly overridden probes must report source=override"
    );
    let xcode_entry = &probe_values[2];
    assert_eq!(
        xcode_entry.value, "Xcode 26.0 Build 17A1\n",
        "whole-output overrides hash the override text plus one trailing newline"
    );
    let target_entry = &probe_values[0];
    assert_eq!(
        target_entry.value, "aarch64-apple-darwin",
        "the target value feeds the manifest raw, without an appended newline"
    );
});

offline_test!(manifest_lines_digest_exactly_the_reported_probe_values, {
    let probe_values = compatibility_probe_values(&overridden_request()).unwrap();
    let manifest_lines = compatibility_manifest_lines(&overridden_request()).unwrap();
    let mut manifest_lines_iter = manifest_lines.lines();
    for probe_entry in &probe_values {
        let manifest_line = manifest_lines_iter.next().unwrap();
        let expected_line = format!(
            "compatibility\t{}\t{}",
            probe_entry.name,
            digest_bytes(probe_entry.value.as_bytes())
        );
        assert_eq!(
            manifest_line, expected_line,
            "the identity manifest must digest exactly the value --explain reports"
        );
    }
    assert_eq!(
        manifest_lines_iter.next(),
        None,
        "the manifest must carry no entries beyond the reported probe values"
    );
});

offline_test!(
    probe_values_preserve_the_digest_semantics_of_composite_probes,
    {
        let probe_values = compatibility_probe_values(&overridden_request()).unwrap();
        let sdk_entry = &probe_values[3];
        assert_eq!(
            sdk_entry.value, "version=15.4\nbuild=24A123\n",
            "composite overrides hash the override text plus one trailing newline"
        );
    }
);

// The one-line helper under test lives in xtask; this crate-level test only
// guards the value shapes the helper flattens, so the CI log format stays
// predictable for flap forensics.
offline_test!(probe_value_shapes_stay_single_line_flattenable, {
    let probe_values: Vec<ProbeValue> = compatibility_probe_values(&overridden_request()).unwrap();
    for probe_entry in &probe_values {
        assert!(
            !probe_entry.value.contains('\r'),
            "probe values must not carry carriage returns: {}",
            probe_entry.name
        );
    }
});
