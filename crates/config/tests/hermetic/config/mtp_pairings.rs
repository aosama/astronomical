use astronomical_config::{AstronomicalConfig, AstronomicalConfigError};

use super::write_config;

#[test]
fn should_return_empty_pairings_by_default() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(temporary_home_directory.path(), r#"{}"#);

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(
        astronomical_config
            .mtp_pairings()
            .expect("pairings should resolve")
            .is_empty()
    );
}

#[test]
fn should_load_a_single_valid_pairing() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP"
            }
          ]
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    let resolved = astronomical_config
        .mtp_pairings()
        .expect("pairings should resolve");
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].target_model_id(), "Qwen-Target");
    assert_eq!(resolved[0].drafter_model_id(), "Qwen-Target-MTP");
}

#[test]
fn should_load_multiple_valid_pairings_with_shared_drafter() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target-A",
              "drafter_model_id": "Shared-MTP"
            },
            {
              "target_model_id": "Qwen-Target-B",
              "drafter_model_id": "Shared-MTP"
            }
          ]
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    let resolved = astronomical_config
        .mtp_pairings()
        .expect("pairings should resolve");
    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved[0].target_model_id(), "Qwen-Target-A");
    assert_eq!(resolved[1].target_model_id(), "Qwen-Target-B");
    assert_eq!(resolved[0].drafter_model_id(), "Shared-MTP");
    assert_eq!(resolved[1].drafter_model_id(), "Shared-MTP");
}

#[test]
fn should_trim_whitespace_from_model_ids() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "  Qwen-Target  ",
              "drafter_model_id": "  Qwen-Target-MTP  "
            }
          ]
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    let resolved = astronomical_config
        .mtp_pairings()
        .expect("pairings should resolve");
    assert_eq!(resolved[0].target_model_id(), "Qwen-Target");
    assert_eq!(resolved[0].drafter_model_id(), "Qwen-Target-MTP");
}

#[test]
fn should_reject_unknown_fields_in_pairing() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP",
              "extra_field": "rejected"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_require_both_pairing_identifiers() {
    for incomplete_pairing in [
        r#"{ "target_model_id": "Qwen-Target" }"#,
        r#"{ "drafter_model_id": "Qwen-Target-MTP" }"#,
    ] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(r#"{{ "mtp_pairings": [{incomplete_pairing}] }}"#),
        );

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::ParseConfigFile { .. })
        ));
    }
}

#[test]
fn should_reject_an_empty_target_model_id() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "",
              "drafter_model_id": "Qwen-Target-MTP"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingTargetModelIdMustNotBeEmpty)
    ));
}

#[test]
fn should_reject_an_empty_drafter_model_id() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": ""
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingDrafterModelIdMustNotBeEmpty)
    ));
}

#[test]
fn should_reject_self_referential_pairing() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingSelfReference { target_model_id })
        if target_model_id == "Qwen-Target"
    ));
}

#[test]
fn should_reject_exact_duplicate_pairing_declarations() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP"
            },
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingDuplicateTarget { target_model_id })
        if target_model_id == "Qwen-Target"
    ));
}

#[test]
fn should_reject_conflicting_target_to_different_drafter_mappings() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Drafter-A"
            },
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Drafter-B"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingConflictingTargetMapping {
            target_model_id,
            drafter_model_id_a,
            drafter_model_id_b,
        }) if target_model_id == "Qwen-Target"
            && drafter_model_id_a == "Drafter-A"
            && drafter_model_id_b == "Drafter-B"
    ));
}

#[test]
fn should_reject_pairing_cycles() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target-A",
              "drafter_model_id": "Qwen-Target-B"
            },
            {
              "target_model_id": "Qwen-Target-B",
              "drafter_model_id": "Qwen-Target-A"
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingCycle { .. })
    ));
}

#[test]
fn should_reject_after_whitespace_trimming() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_pairings": [
            {
              "target_model_id": "  Qwen-Target  ",
              "drafter_model_id": "  Qwen-Target  "
            }
          ]
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::MtpPairingSelfReference { .. })
    ));
}

#[test]
fn should_allow_pairings_with_mtp_enabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_enabled": true,
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP"
            }
          ]
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("pairings should be composable with mtp_enabled");

    assert!(astronomical_config.mtp_enabled());
    let resolved = astronomical_config
        .mtp_pairings()
        .expect("pairings should resolve");
    assert_eq!(resolved.len(), 1);
}

#[test]
fn should_allow_pairings_with_mtp_disabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_enabled": false,
          "mtp_pairings": [
            {
              "target_model_id": "Qwen-Target",
              "drafter_model_id": "Qwen-Target-MTP"
            }
          ]
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("pairings should be composable with mtp_enabled=false");

    assert!(!astronomical_config.mtp_enabled());
    let resolved = astronomical_config
        .mtp_pairings()
        .expect("pairings should resolve");
    assert_eq!(resolved.len(), 1);
}
