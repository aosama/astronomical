use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use astronomical_native_identity::digest_bytes;

use super::thin_talk_assets::{
    PinnedAsset, VENDOR_DIRECTORY, asset_problem, ensure_asset, every_pinned_asset,
    partial_download_path, run,
};

#[test]
fn should_pin_every_asset_once_with_a_lowercase_sha256() {
    let mut vendored_names = BTreeSet::new();
    for pinned_asset in every_pinned_asset() {
        assert!(
            vendored_names.insert(pinned_asset.vendored_name),
            "{} is pinned twice",
            pinned_asset.vendored_name
        );
        assert!(
            !pinned_asset.package_path.starts_with('/'),
            "{}",
            pinned_asset.package_path
        );
        if let Some(pinned_digest) = pinned_asset.sha256 {
            assert!(
                astronomical_lowercase_hex::is_lowercase_hex_of_length(pinned_digest, 64),
                "{}",
                pinned_asset.vendored_name
            );
        }
    }
    assert_eq!(vendored_names.len(), 34);
}

/// The committed vendor directory is the offline canvas's source of truth;
/// verify-only must pass on it without touching the network.
#[test]
fn should_verify_the_committed_vendor_directory_without_downloading() {
    let repository_root = crate::workspace_root_from_manifest_directory();

    assert_eq!(run(&repository_root, &["--verify-only".to_owned()]), Ok(()));
}

#[test]
fn should_report_missing_empty_and_mismatched_assets() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let asset_path = fixture_directory.path().join("asset.js");
    let pinned_asset = PinnedAsset {
        vendored_name: "asset.js",
        package_path: "pkg@1/asset.js",
        sha256: Some("0000"),
    };
    let license = PinnedAsset {
        vendored_name: "LICENSE.txt",
        package_path: "pkg@1/LICENSE",
        sha256: None,
    };

    assert!(
        asset_problem(&asset_path, &pinned_asset)
            .expect("missing")
            .contains("missing")
    );
    std::fs::write(&asset_path, "").expect("empty");
    assert!(
        asset_problem(&asset_path, &pinned_asset)
            .expect("empty")
            .contains("empty")
    );
    std::fs::write(&asset_path, "bytes").expect("content");
    assert!(
        asset_problem(&asset_path, &pinned_asset)
            .expect("mismatch")
            .contains("digest mismatch")
    );
    assert!(asset_problem(&fixture_directory.path().join("LICENSE.txt"), &license).is_some());
}

#[test]
fn should_accept_a_matching_asset_and_a_present_license_in_verify_only_mode() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    std::fs::write(fixture_directory.path().join("asset.js"), "bytes").expect("asset");
    std::fs::write(fixture_directory.path().join("LICENSE.txt"), "terms").expect("license");
    let digest = digest_bytes(b"bytes");
    let pinned_asset = PinnedAsset {
        vendored_name: "asset.js",
        package_path: "pkg@1/asset.js",
        sha256: Some(Box::leak(digest.into_boxed_str())),
    };
    let license = PinnedAsset {
        vendored_name: "LICENSE.txt",
        package_path: "pkg@1/LICENSE",
        sha256: None,
    };

    assert_eq!(
        ensure_asset(fixture_directory.path(), &pinned_asset, true),
        Ok(())
    );
    assert_eq!(
        ensure_asset(fixture_directory.path(), &license, true),
        Ok(())
    );
    assert!(
        ensure_asset(
            fixture_directory.path(),
            &PinnedAsset {
                vendored_name: "absent.js",
                ..pinned_asset
            },
            true
        )
        .is_err()
    );
}

#[test]
fn should_download_beside_the_asset_and_refuse_unknown_flags() {
    assert_eq!(
        partial_download_path(Path::new("/vendor/katex/katex.min.js")),
        PathBuf::from("/vendor/katex/katex.min.js.download")
    );
    assert!(run(Path::new("/repository"), &["--force".to_owned()]).is_err());
    assert!(VENDOR_DIRECTORY.ends_with("Resources/web/vendor"));
}
