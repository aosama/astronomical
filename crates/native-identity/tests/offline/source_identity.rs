//! Source-only identity over the real repository tree: the git enumeration
//! seam is the subject, so this stays an offline integration journey rather
//! than a beside-source unit test (real filesystem, no network, no model
//! artifacts).

use astronomical_native_identity::probes::CompatibilityOverrides;
use astronomical_native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};
use astronomical_test_harness::offline_test;

fn repository_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is <workspace-root>/crates/native-identity; the
    // source-identity pathspecs are workspace-relative.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

offline_test!(source_only_identity_is_stable_and_self_covering, {
    let repository_root = repository_root();
    let first_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: CompatibilityOverrides::none(),
    })
    .unwrap();
    let second_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: CompatibilityOverrides::none(),
    })
    .unwrap();
    assert_eq!(first_identity, second_identity);
    assert_eq!(first_identity.len(), 64);
    assert!(
        first_identity
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    );
});

offline_test!(profile_choice_changes_the_identity, {
    let repository_root = repository_root();
    let core_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: CompatibilityOverrides::none(),
    })
    .unwrap();
    let memory_contract_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::CoreMemoryContract,
        scope: IdentityScope::SourceOnly,
        overrides: CompatibilityOverrides::none(),
    })
    .unwrap();
    assert_ne!(core_identity, memory_contract_identity);
});
