//! Contracts of the store validate-and-upload command that need no Apple
//! account: what Apple's tool is asked, which packages are refused before
//! anything is sent, and where the private key is looked for.

use std::path::{Path, PathBuf};

use super::{
    AppStoreUploadRequest, StoreSubmissionStep, missing_private_key_message,
    private_key_candidates, store_package_signature_problem, submission_arguments,
    validate_and_upload_for_app_store,
};

fn sample_upload_request(should_upload: bool) -> AppStoreUploadRequest {
    AppStoreUploadRequest {
        package: PathBuf::from("/builds/Astronomical-0.3.0-app-store.pkg"),
        api_key_identifier: "KEYFIXTURE".to_owned(),
        api_issuer_identifier: "00000000-0000-0000-0000-000000000000".to_owned(),
        should_upload,
    }
}

#[test]
fn should_ask_apple_to_validate_with_the_key_and_issuer() {
    assert_eq!(
        submission_arguments(&sample_upload_request(false), StoreSubmissionStep::Validate),
        [
            "altool",
            "--validate-app",
            "/builds/Astronomical-0.3.0-app-store.pkg",
            "--api-key",
            "KEYFIXTURE",
            "--api-issuer",
            "00000000-0000-0000-0000-000000000000",
        ]
    );
}

#[test]
fn should_ask_apple_to_upload_the_same_package_with_the_same_credentials() {
    let validation_arguments =
        submission_arguments(&sample_upload_request(true), StoreSubmissionStep::Validate);
    let upload_arguments =
        submission_arguments(&sample_upload_request(true), StoreSubmissionStep::Upload);

    assert_eq!(upload_arguments[1], "--upload-package");
    assert_eq!(upload_arguments[2..], validation_arguments[2..]);
}

#[test]
fn should_accept_only_a_package_signed_with_the_store_installer_certificate() {
    let store_signature_report = "Status: signed by a developer certificate issued by Apple (Development)\n   1. 3rd Party Mac Developer Installer: Example Corp (TEAM123456)\n";
    let direct_signature_report = "Status: signed by a developer certificate issued by Apple for distribution\n   1. Developer ID Installer: Example Corp (TEAM123456)\n";
    let unsigned_report = "Status: no signature\n";

    assert_eq!(
        store_package_signature_problem(store_signature_report),
        None
    );
    for refused_report in [direct_signature_report, unsigned_report, ""] {
        let signature_problem = store_package_signature_problem(refused_report)
            .expect("only a store installer signature is accepted");
        assert!(
            signature_problem.contains("cargo xtask release app-store-pkg"),
            "{signature_problem}"
        );
    }
}

#[test]
fn should_look_for_the_private_key_where_apples_tool_does() {
    assert_eq!(
        private_key_candidates(Path::new("/Users/example"), "KEYFIXTURE"),
        [
            PathBuf::from("/Users/example/.appstoreconnect/private_keys/AuthKey_KEYFIXTURE.p8"),
            PathBuf::from("/Users/example/.private_keys/AuthKey_KEYFIXTURE.p8"),
            PathBuf::from("/Users/example/private_keys/AuthKey_KEYFIXTURE.p8"),
        ]
    );
}

#[test]
fn should_say_where_a_missing_private_key_belongs() {
    let missing_key_message = missing_private_key_message("KEYFIXTURE");

    assert!(
        missing_key_message.contains("~/.appstoreconnect/private_keys/AuthKey_KEYFIXTURE.p8"),
        "{missing_key_message}"
    );
}

#[test]
fn should_refuse_a_package_that_does_not_exist_before_contacting_apple() {
    let mut absent_package_request = sample_upload_request(true);
    absent_package_request.package = PathBuf::from("/nowhere/absent-app-store.pkg");

    let refusal = validate_and_upload_for_app_store(&absent_package_request)
        .expect_err("an absent package cannot be validated");

    assert!(refusal.starts_with("store package not found"), "{refusal}");
}

/// A file that is not an installer package at all carries no store
/// signature, so it is refused locally and nothing is sent.
#[test]
fn should_refuse_a_file_that_is_not_a_signed_store_package() {
    let fixture_directory = tempfile::tempdir().expect("a temporary directory");
    let unsigned_package = fixture_directory.path().join("not-a-package.pkg");
    std::fs::write(&unsigned_package, "not an installer package").expect("fixture file");
    let mut unsigned_package_request = sample_upload_request(true);
    unsigned_package_request.package = unsigned_package;

    let refusal = validate_and_upload_for_app_store(&unsigned_package_request)
        .expect_err("an unsigned file is not a store package");

    assert!(
        refusal.starts_with("this is not a store package"),
        "{refusal}"
    );
}
