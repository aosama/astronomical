use super::native_error_text;

#[test]
fn should_read_the_native_error_message_up_to_its_terminator() {
    let mut native_error = [0 as std::ffi::c_char; 8];
    for (byte_index, byte) in b"bad".iter().enumerate() {
        native_error[byte_index] = *byte as std::ffi::c_char;
    }

    assert_eq!(native_error_text(&native_error), "bad");
}
