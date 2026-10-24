use crate::probes::ToolchainOverrides;
use crate::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, NativeIdentityError,
    SOURCE_IDENTITY_PATHSPECS, digest_bytes, enumerate_tracked_source_paths, native_build_identity,
    native_build_identity_with_probe_values,
};

/// `CARGO_MANIFEST_DIR` is `<workspace-root>/crates/native-identity`; the
/// source-identity pathspecs are workspace-relative, so beside-source tests
/// resolve the workspace root rather than relying on the working directory.
fn workspace_root_from_manifest_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lives two levels below the workspace root")
        .to_path_buf()
}

fn fixture_directory(fixture_name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from("target")
        .join("native-identity-fixtures")
        .join(format!(
            "{fixture_name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("the system clock should be after the Unix epoch")
                .as_nanos()
        ))
}

#[test]
fn should_round_trip_both_supported_profile_spellings() {
    assert_eq!(
        NativeBuildProfile::from_profile_text("core"),
        Ok(NativeBuildProfile::Core)
    );
    assert_eq!(
        NativeBuildProfile::from_profile_text("core+memory-contract"),
        Ok(NativeBuildProfile::CoreMemoryContract)
    );
}

#[test]
fn should_reject_unknown_profile_spellings() {
    assert_eq!(
        NativeBuildProfile::from_profile_text("release"),
        Err(NativeIdentityError::UnsupportedProfile(
            "release".to_owned()
        ))
    );
}

#[test]
fn should_cover_native_inputs_and_exclude_executors_from_source_identity() {
    let covered: Vec<&str> = SOURCE_IDENTITY_PATHSPECS.to_vec();
    assert!(
        covered.contains(&"crates/native-identity/src"),
        "the identity policy must self-cover its own sources"
    );
    assert!(
        covered.contains(&"third-party/pins") && covered.contains(&"third-party/patches"),
        "third-party pin and patch changes must change the identity"
    );
    let joined_policy = covered.join("\n");
    assert!(
        !joined_policy.contains("build.rs") && !joined_policy.contains("xtask"),
        "executors are deliberately excluded so orchestration edits never force a CMake rebuild"
    );
}

#[test]
fn should_match_reference_sha256_vectors() {
    // sha256("") and sha256("abc") reference digests from FIPS 180-4 test
    // vectors; they pin the hex-encoding and hashing primitive together.
    assert_eq!(
        digest_bytes(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        digest_bytes(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn should_require_a_target_triple_for_full_scope() {
    let repository_root = workspace_root_from_manifest_directory();
    let failure = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: ToolchainOverrides::none(),
    })
    .unwrap_err();
    assert!(matches!(failure, NativeIdentityError::MissingTargetTriple));
}

#[test]
fn should_reject_non_release_build_types_for_full_scope() {
    let repository_root = workspace_root_from_manifest_directory();
    let failure = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: ToolchainOverrides {
            target_triple: Some("aarch64-apple-darwin".to_owned()),
            build_type: Some("Debug".to_owned()),
            ..ToolchainOverrides::none()
        },
    })
    .unwrap_err();
    assert!(matches!(
        failure,
        NativeIdentityError::UnsupportedBuildType(build_type) if build_type == "Debug"
    ));
}

#[test]
fn should_keep_direct_toolchain_fixtures_offline_deterministic() {
    // Every toolchain probe is overridden, so no real tool is invoked and the
    // identity is a pure function of the inputs: identical requests must
    // agree, and a changed override must change the identity.
    let repository_root = workspace_root_from_manifest_directory();
    let overridden = ToolchainOverrides {
        xcode: Some("Xcode 26.0".to_owned()),
        sdk: Some("version=15.4\nbuild=24A123".to_owned()),
        clang: Some("version=17.0.0\ntarget=arm64-apple-darwin24".to_owned()),
        cmake: Some("cmake version 4.2.1".to_owned()),
        rustc: Some("rustc 1.97.1".to_owned()),
        target_triple: Some("aarch64-apple-darwin".to_owned()),
        build_type: None,
    };
    let first_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: overridden.clone(),
    })
    .unwrap();
    let second_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: overridden.clone(),
    })
    .unwrap();
    assert_eq!(first_identity, second_identity);
    assert_eq!(first_identity.len(), 64);

    let mut changed_override = overridden.clone();
    changed_override.rustc = Some("rustc 1.98.0".to_owned());
    let changed_identity = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: changed_override,
    })
    .unwrap();
    assert_ne!(first_identity, changed_identity);
}

#[test]
fn should_reject_enumeration_outside_a_git_worktree() {
    let fixture_root = fixture_directory("not-git");
    std::fs::create_dir_all(&fixture_root).unwrap();
    std::fs::write(fixture_root.join(".git"), "gitdir: missing-repository\n").unwrap();
    let failure = enumerate_tracked_source_paths(&fixture_root);
    std::fs::remove_dir_all(&fixture_root).unwrap();
    assert!(matches!(
        failure,
        Err(NativeIdentityError::GitEnumeration(_))
    ));
}

#[test]
fn should_ignore_untracked_files_under_covered_paths() {
    let fixture_root = fixture_directory("untracked");
    std::fs::create_dir_all(fixture_root.join("crates/native-identity/src")).unwrap();
    let git = |arguments: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(&fixture_root)
            .args(arguments)
            .output()
            .unwrap()
    };
    assert!(git(&["init", "--quiet"]).status.success());
    std::fs::write(
        fixture_root.join("crates/native-identity/src/lib.rs"),
        "tracked identity source\n",
    )
    .unwrap();
    assert!(git(&["add", "."]).status.success());
    let tracked_enumeration = enumerate_tracked_source_paths(&fixture_root).unwrap();
    std::fs::write(
        fixture_root.join("crates/native-identity/src/scratch_file.rs"),
        "untracked scratch that must not churn the identity\n",
    )
    .unwrap();
    let enumeration_with_scratch = enumerate_tracked_source_paths(&fixture_root).unwrap();
    assert_eq!(
        tracked_enumeration, enumeration_with_scratch,
        "an untracked scratch file under a covered path must not enter the identity"
    );

    std::fs::remove_dir_all(&fixture_root).unwrap();
}

#[test]
fn should_match_the_single_pass_identity_to_the_shared_identity_entry_point() {
    let repository_root = workspace_root_from_manifest_directory();
    let request = FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: ToolchainOverrides {
            xcode: Some("Xcode 26.0".to_owned()),
            sdk: Some("version=15.4\nbuild=24A123".to_owned()),
            clang: Some("version=17.0.0\ntarget=arm64-apple-darwin24".to_owned()),
            cmake: Some("cmake version 4.2.1".to_owned()),
            rustc: Some("rustc 1.97.1".to_owned()),
            target_triple: Some("aarch64-apple-darwin".to_owned()),
            build_type: None,
        },
    };
    let (identity, probe_values) = native_build_identity_with_probe_values(&request).unwrap();
    assert_eq!(
        identity,
        native_build_identity(&request).unwrap(),
        "the single-pass entry point must hash identically to the CLI face"
    );
    assert_eq!(probe_values.len(), 7);
}
