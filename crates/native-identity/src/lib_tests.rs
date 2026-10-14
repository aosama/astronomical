use astronomical_test_harness::offline_test;

use crate::probes::CompatibilityOverrides;
use crate::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, NativeIdentityError,
    SOURCE_IDENTITY_PATHSPECS, digest_bytes, enumerate_tracked_source_paths, native_build_identity,
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

offline_test!(profile_text_round_trips_for_both_supported_profiles, {
    assert_eq!(
        NativeBuildProfile::from_profile_text("core"),
        Ok(NativeBuildProfile::Core)
    );
    assert_eq!(
        NativeBuildProfile::from_profile_text("core+memory-contract"),
        Ok(NativeBuildProfile::CoreMemoryContract)
    );
});

offline_test!(profile_text_rejects_unknown_spellings, {
    assert_eq!(
        NativeBuildProfile::from_profile_text("release"),
        Err(NativeIdentityError::UnsupportedProfile(
            "release".to_owned()
        ))
    );
});

offline_test!(
    source_identity_policy_covers_native_inputs_and_excludes_executors,
    {
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
);

offline_test!(digest_bytes_matches_reference_sha256_vectors, {
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
});

offline_test!(full_scope_requires_a_target_triple, {
    let repository_root = workspace_root_from_manifest_directory();
    let failure = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: CompatibilityOverrides::none(),
    })
    .unwrap_err();
    assert!(matches!(failure, NativeIdentityError::MissingTargetTriple));
});

offline_test!(full_scope_rejects_non_release_build_types, {
    let repository_root = workspace_root_from_manifest_directory();
    let failure = native_build_identity(&FingerprintRequest {
        repository_root: &repository_root,
        profile: NativeBuildProfile::Core,
        scope: IdentityScope::Full,
        overrides: CompatibilityOverrides {
            target_triple: Some("aarch64-apple-darwin".to_owned()),
            build_type: Some("Debug".to_owned()),
            ..CompatibilityOverrides::none()
        },
    })
    .unwrap_err();
    assert!(matches!(
        failure,
        NativeIdentityError::UnsupportedBuildType(build_type) if build_type == "Debug"
    ));
});

offline_test!(
    full_scope_with_total_overrides_stays_offline_deterministic,
    {
        // Every toolchain probe is overridden, so no real tool is invoked and the
        // identity is a pure function of the inputs: identical requests must
        // agree, and a changed override must change the identity.
        let repository_root = workspace_root_from_manifest_directory();
        let overridden = CompatibilityOverrides {
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
);

offline_test!(enumeration_fails_loudly_outside_a_git_worktree, {
    let temporary_directory = std::env::temp_dir().join(format!(
        "astronomical-native-identity-not-git-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&temporary_directory).unwrap();
    let failure = enumerate_tracked_source_paths(&temporary_directory);
    std::fs::remove_dir_all(&temporary_directory).unwrap();
    assert!(matches!(
        failure,
        Err(NativeIdentityError::GitEnumeration(_))
    ));
});
