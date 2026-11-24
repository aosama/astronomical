use super::super::family_test_support::speech_configuration_naming;
use super::{FAMILY_MISMATCH_REASON, ensure_configuration_names_classified_family};

#[test]
fn should_accept_a_policy_naming_the_family_found_on_disk() {
    assert_eq!(
        ensure_configuration_names_classified_family(
            Some(config::PHONON_MODEL_FAMILY),
            &speech_configuration_naming("phonon"),
        ),
        Ok(())
    );
}

#[test]
fn should_refuse_a_policy_naming_another_family_of_the_same_kind() {
    assert_eq!(
        ensure_configuration_names_classified_family(
            Some(config::KOKORO_MODEL_FAMILY),
            &speech_configuration_naming("phonon"),
        ),
        Err(FAMILY_MISMATCH_REASON.to_owned())
    );
}

#[test]
fn should_leave_an_unclassified_folder_to_the_factory_refusal() {
    assert_eq!(
        ensure_configuration_names_classified_family(None, &speech_configuration_naming("kokoro")),
        Ok(()),
        "a folder of no known family gets the factory's own unsupported-family refusal"
    );
}
