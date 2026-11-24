//! Source-only identity over the real repository tree: the git enumeration
//! seam is the subject, so this stays an integration journey rather
//! than a beside-source unit test (real filesystem, no network, no model
//! artifacts).

use native_identity::probes::ToolchainOverrides;
use native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};

fn repository_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR lies below the workspace root, which holds Cargo.lock; the
    // source-identity pathspecs are workspace-relative.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|ancestor_directory| ancestor_directory.join("Cargo.lock").is_file())
        .expect("the crate lives inside the workspace, beside its Cargo.lock")
        .to_path_buf()
}

#[test]
fn should_keep_source_only_identity_stable_and_self_covering() {
    let repository_root = repository_root();
    let first_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: ToolchainOverrides::none(),
    })
    .unwrap();
    let second_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: ToolchainOverrides::none(),
    })
    .unwrap();
    assert_eq!(first_identity, second_identity);
    assert_eq!(first_identity.len(), 64);
    assert!(
        first_identity
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    );
}

#[test]
fn should_change_identity_when_the_profile_changes() {
    let repository_root = repository_root();
    let core_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::SourceOnly,
        overrides: ToolchainOverrides::none(),
    })
    .unwrap();
    let memory_contract_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::CoreMemoryContract,
        scope: IdentityScope::SourceOnly,
        overrides: ToolchainOverrides::none(),
    })
    .unwrap();
    assert_ne!(core_identity, memory_contract_identity);
}
