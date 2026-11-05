use super::{append_bytes_field, append_string_field, append_varint, append_varint_field};

fn encoded_varint(unsigned_value: u64) -> Vec<u8> {
    let mut document = Vec::new();
    append_varint(&mut document, unsigned_value);
    document
}

#[test]
fn should_encode_single_byte_varints() {
    assert_eq!(encoded_varint(0), vec![0x00]);
    assert_eq!(encoded_varint(127), vec![0x7f]);
}

#[test]
fn should_continue_a_varint_across_the_seven_bit_boundary() {
    assert_eq!(encoded_varint(128), vec![0x80, 0x01]);
    assert_eq!(encoded_varint(300), vec![0xac, 0x02]);
}

#[test]
fn should_encode_the_largest_varint_in_ten_bytes() {
    let largest = encoded_varint(u64::MAX);

    assert_eq!(largest.len(), 10);
    assert_eq!(largest[9], 0x01);
}

#[test]
fn should_prefix_fields_with_their_number_and_wire_type() {
    let mut document = Vec::new();
    append_varint_field(&mut document, 1, 7);
    append_string_field(&mut document, 2, "ab");
    append_bytes_field(&mut document, 500, &[]);

    assert_eq!(
        document,
        vec![
            0x08, 0x07, // field 1, varint 7
            0x12, 0x02, b'a', b'b', // field 2, two bytes
            0xa2, 0x1f, 0x00, // field 500, empty
        ]
    );
}
