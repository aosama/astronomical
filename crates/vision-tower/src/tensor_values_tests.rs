use super::*;

fn half(bits: u16) -> f32 {
    decode_float_values("F16", &bits.to_le_bytes()).unwrap()[0]
}

#[test]
fn should_decode_each_float_width() {
    let bfloat16_one = ((1.5_f32.to_bits() >> 16) as u16).to_le_bytes();
    assert_eq!(decode_float_values("BF16", &bfloat16_one).unwrap(), [1.5]);
    assert_eq!(
        decode_float_values("F32", &2.5_f32.to_le_bytes()).unwrap(),
        [2.5]
    );
    assert_eq!(
        decode_float_values("F64", &(-0.5_f64).to_le_bytes()).unwrap(),
        [-0.5]
    );
}

#[test]
fn should_decode_half_precision_across_its_edge_cases() {
    assert_eq!(half(0x3c00), 1.0);
    assert_eq!(half(0xc000), -2.0);
    assert_eq!(half(0x0000), 0.0);
    assert!(half(0x8000).is_sign_negative());
    assert_eq!(half(0x7c00), f32::INFINITY);
    assert_eq!(half(0xfc00), f32::NEG_INFINITY);
    assert!(half(0x7e00).is_nan());
    assert_eq!(half(0x0001), 2.0_f32.powi(-24), "the smallest subnormal");
    assert_eq!(half(0x0400), 2.0_f32.powi(-14), "the smallest normal");
    assert_eq!(
        half(0x03ff),
        2.0_f32.powi(-14) * (1023.0 / 1024.0),
        "the largest subnormal"
    );
}

#[test]
fn should_refuse_non_floats_and_partial_elements() {
    assert_eq!(decode_float_values("U32", &[0; 4]), None);
    assert_eq!(decode_float_values("BF16", &[0; 3]), None);
    assert_eq!(decode_float_values("F32", &[]).unwrap(), Vec::<f32>::new());
    assert!(is_float_data_type("BF16"));
    assert!(!is_float_data_type("U32"));
}
