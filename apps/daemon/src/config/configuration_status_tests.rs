//! Beside-source unit contracts for the chunking summary value builders in
//! `configuration_status.rs`. Every reported chunking field flows through the
//! same three-slot rule — configured only when the exact field's presence flag
//! is set, default always present for numeric fields, effective regardless of
//! flags — so those rules are pinned here with plain values, no worker or
//! model runtime needed.

use super::{
    ConfigurationValue, NullableConfigurationValue, reported_chunking_configuration_value,
    reported_nullable_chunking_configuration_value,
};

#[test]
fn should_report_the_configured_value_only_when_its_field_flag_is_set() {
    let configured_field =
        reported_chunking_configuration_value(true, Some(2048_u32), 512_u32, Some(2048_u32));

    assert_eq!(configured_field.configured, Some(2048));
    assert_eq!(configured_field.default, Some(512));
    assert_eq!(configured_field.effective, Some(2048));
}

#[test]
fn should_report_no_configured_value_when_the_field_flag_is_clear() {
    // A chunking block present in configuration but without this exact field
    // must not borrow a sibling field's value as "configured".
    let unconfigured_field =
        reported_chunking_configuration_value(false, Some(2048_u32), 512_u32, Some(512_u32));

    assert_eq!(unconfigured_field.configured, None);
    assert_eq!(unconfigured_field.default, Some(512));
    assert_eq!(unconfigured_field.effective, Some(512));
}

#[test]
fn should_report_the_effective_value_even_when_nothing_was_configured() {
    // A worker started with purely default configuration still reports the
    // value it actually resolved, alongside the built-in default.
    let defaulted_field =
        reported_chunking_configuration_value(false, None, 512_u32, Some(512_u32));

    assert_eq!(defaulted_field.configured, None);
    assert_eq!(defaulted_field.default, Some(512));
    assert_eq!(defaulted_field.effective, Some(512));
}

#[test]
fn should_report_a_nullable_configured_value_only_when_its_field_flag_is_set() {
    let configured_nullable =
        reported_nullable_chunking_configuration_value(true, Some(256_u32), Some(256_u32));

    assert!(configured_nullable.is_configured);
    assert_eq!(configured_nullable.configured, Some(256));
    assert_eq!(configured_nullable.effective, Some(256));
}

#[test]
fn should_keep_a_cleared_nullable_field_marked_configured_but_empty() {
    // The operator explicitly cleared the field: the presence flag stays true
    // while the configured slot reports none, distinguishing an explicit clear
    // from a field that was never mentioned.
    let cleared_nullable: NullableConfigurationValue<u32> =
        reported_nullable_chunking_configuration_value(true, None, None);

    assert!(cleared_nullable.is_configured);
    assert_eq!(cleared_nullable.configured, None);
    assert_eq!(cleared_nullable.effective, None);
}

#[test]
fn should_leave_the_nullable_default_slot_empty_when_unmentioned() {
    let unmentioned_nullable =
        reported_nullable_chunking_configuration_value(false, Some(256_u32), Some(256_u32));

    assert!(!unmentioned_nullable.is_configured);
    assert_eq!(unmentioned_nullable.configured, None);
    assert_eq!(unmentioned_nullable.default, None);
    assert_eq!(unmentioned_nullable.effective, Some(256));
}

#[test]
fn should_type_check_the_value_builders_as_pure_constructors() {
    // Boundary guard: both builders are pure constructors over Copy values, so
    // the same inputs must always produce the same summary.
    let first = reported_chunking_configuration_value(true, Some(1_u16), 2_u16, Some(3_u16));
    let second = reported_chunking_configuration_value(true, Some(1_u16), 2_u16, Some(3_u16));
    let ConfigurationValue {
        configured: first_configured,
        default: first_default,
        effective: first_effective,
    } = first;
    let ConfigurationValue {
        configured: second_configured,
        default: second_default,
        effective: second_effective,
    } = second;

    assert_eq!(first_configured, second_configured);
    assert_eq!(first_default, second_default);
    assert_eq!(first_effective, second_effective);
}
