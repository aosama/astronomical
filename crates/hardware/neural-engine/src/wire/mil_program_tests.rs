use super::{
    MilDataType, MilOperation, append_map_entry, blob_file_tensor_value, bool_scalar_value,
    int32_tensor_value, named_value_type, string_scalar_value, tensor_value_type,
};

fn contains_subsequence(encoded: &[u8], expected_bytes: &[u8]) -> bool {
    for window in encoded.windows(expected_bytes.len()) {
        if window == expected_bytes {
            return true;
        }
    }
    false
}

#[test]
fn should_encode_a_rank_four_fp16_tensor_type() {
    let value_type = tensor_value_type(MilDataType::Float16, &[1, 4, 1, 2]);

    // ValueType{tensorType{dataType 10, rank 4, four constant dimensions}}.
    assert_eq!(
        value_type,
        vec![
            0x0a, 0x1c, 0x08, 0x0a, 0x10, 0x04, 0x1a, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x1a, 0x04,
            0x0a, 0x02, 0x08, 0x04, 0x1a, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x1a, 0x04, 0x0a, 0x02,
            0x08, 0x02,
        ]
    );
}

#[test]
fn should_encode_a_scalar_as_rank_zero_without_dimensions() {
    assert_eq!(
        tensor_value_type(MilDataType::Int32, &[]),
        vec![0x0a, 0x04, 0x08, 0x17, 0x10, 0x00]
    );
}

#[test]
fn should_pack_int32_values_as_varints() {
    let value = int32_tensor_value(&[1, 300], &[2]);

    // RepeatedInts{values: packed [1, 300]} inside TensorValue field 2.
    assert!(contains_subsequence(
        &value,
        &[0x12, 0x05, 0x0a, 0x03, 0x01, 0xac, 0x02]
    ));
}

#[test]
fn should_sign_extend_a_negative_int32() {
    let value = int32_tensor_value(&[-1], &[]);

    assert!(contains_subsequence(
        &value,
        &[
            0x0a, 0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01
        ]
    ));
}

#[test]
fn should_encode_a_string_scalar_in_the_strings_field() {
    let value = string_scalar_value("valid");

    assert!(contains_subsequence(&value, &[0x22, 0x07, 0x0a, 0x05]));
    assert!(contains_subsequence(&value, b"valid"));
}

#[test]
fn should_encode_a_bool_scalar_in_the_bools_field_with_its_own_type() {
    let true_value = bool_scalar_value(true);
    let false_value = bool_scalar_value(false);

    // RepeatedBools{values: packed [v]} inside TensorValue field 3.
    assert!(contains_subsequence(
        &true_value,
        &[0x1a, 0x03, 0x0a, 0x01, 0x01]
    ));
    assert!(contains_subsequence(
        &false_value,
        &[0x1a, 0x03, 0x0a, 0x01, 0x00]
    ));
    assert!(
        contains_subsequence(&true_value, &[0x08, 0x01]),
        "BOOL data type"
    );
}

#[test]
fn should_point_a_blob_value_at_its_metadata_offset() {
    let value = blob_file_tensor_value(
        MilDataType::UnsignedInt4,
        &[2, 2],
        "@model_path/weights/weight.bin",
        64,
    );

    // BlobFileValue (Value field 5): fileName then offset 64.
    assert!(contains_subsequence(
        &value,
        b"@model_path/weights/weight.bin"
    ));
    assert!(contains_subsequence(&value, &[0x10, 0x40]));
    assert!(
        contains_subsequence(&value, &[0x08, 0x23]),
        "UINT4 data type"
    );
}

#[test]
fn should_write_a_map_entry_with_key_then_value() {
    let mut message = Vec::new();

    append_map_entry(&mut message, 2, "x", &[0xaa]);

    assert_eq!(
        message,
        vec![0x12, 0x06, 0x0a, 0x01, b'x', 0x12, 0x01, 0xaa]
    );
}

#[test]
fn should_name_the_output_and_its_type() {
    let named_type = named_value_type("y", &[0x0a, 0x00]);

    assert_eq!(named_type, vec![0x0a, 0x01, b'y', 0x12, 0x02, 0x0a, 0x00]);
}

#[test]
fn should_carry_type_name_inputs_and_outputs_in_an_operation() {
    let operation = MilOperation::new("add", "sum_0")
        .input_named("x", "left")
        .input_value("y", &string_scalar_value("ignored"))
        .output("sum_0", &tensor_value_type(MilDataType::Float16, &[1]))
        .into_bytes();

    assert_eq!(&operation[..5], &[0x0a, 0x03, b'a', b'd', b'd']);
    assert!(contains_subsequence(&operation, b"sum_0"));
    // Argument{Binding{name "left"}} under the "x" key.
    assert!(contains_subsequence(
        &operation,
        &[0x0a, 0x06, 0x0a, 0x04, b'l', b'e', b'f', b't']
    ));
}

#[test]
fn should_bind_every_named_value_under_one_list_argument() {
    let operation = MilOperation::new("concat", "joined")
        .input_named_list("values", &["a".to_owned(), "bc".to_owned()])
        .into_bytes();

    // Argument{Binding{name "a"}, Binding{name "bc"}} under the "values" key.
    assert!(contains_subsequence(
        &operation,
        &[
            0x0a, 0x03, 0x0a, 0x01, b'a', 0x0a, 0x04, 0x0a, 0x02, b'b', b'c'
        ]
    ));
}

#[test]
fn should_keep_the_key_of_an_empty_named_list() {
    let operation = MilOperation::new("concat", "joined")
        .input_named_list("values", &[])
        .into_bytes();

    assert!(contains_subsequence(&operation, b"values"));
}
