use super::float_define;

#[test]
fn should_write_a_float_as_the_launchers_always_wrote_it() {
    let rms_norm_epsilon = float_define("RMS_NORM_EPSILON", 1e-6);
    assert_eq!(rms_norm_epsilon.name, "RMS_NORM_EPSILON");
    assert_eq!(rms_norm_epsilon.value, "9.999999975e-7f");
}

#[test]
fn should_write_an_exact_power_of_two_without_rounding() {
    assert_eq!(
        float_define("INVERSE_LANE_COUNT", 0.25).value,
        "2.500000000e-1f"
    );
}

#[test]
fn should_write_a_negative_value_with_its_sign() {
    assert_eq!(float_define("SHIFT", -1.5).value, "-1.500000000e0f");
}
