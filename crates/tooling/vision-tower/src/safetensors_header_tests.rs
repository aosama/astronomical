use super::*;

const FILE: &str = "weights.safetensors";

#[test]
fn should_read_tensors_and_skip_the_metadata_entry() {
    let header = br#"{"__metadata__": {"format": "mlx"},
        "a.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "b.weight": {"dtype": "U32", "shape": [1], "data_offsets": [12, 16]}}"#;
    let entries = parse_header(FILE, header, None).unwrap();
    assert_eq!(entries.len(), 2);
    let first = &entries["a.weight"];
    assert_eq!(first.data_start, 8 + header.len() as u64);
    assert_eq!(first.byte_count, 12);
    assert_eq!(first.element_count(), 6);
    assert_eq!(entries["b.weight"].data_start, 8 + header.len() as u64 + 12);
}

#[test]
fn should_treat_a_scalar_as_one_element() {
    let header = br#"{"a": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}"#;
    assert_eq!(
        parse_header(FILE, header, None).unwrap()["a"].element_count(),
        1
    );
}

#[test]
fn should_accept_a_header_with_only_metadata() {
    assert!(
        parse_header(FILE, br#"{"__metadata__": {}}"#, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn should_accept_an_unknown_data_type_without_checking_its_size() {
    let header = br#"{"a": {"dtype": "C64", "shape": [2], "data_offsets": [0, 3]}}"#;
    assert_eq!(parse_header(FILE, header, None).unwrap()["a"].byte_count, 3);
}

#[test]
fn should_refuse_inconsistent_or_hostile_entries() {
    for hostile_header in [
        &br#"not json"#[..],
        br#"{"a": {"dtype": "BF16"}}"#,
        br#"{"a": {"dtype": "BF16", "shape": [1], "data_offsets": [4, 2]}}"#,
        br#"{"a": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 2]}}"#,
        br#"{"a": {"dtype": "BF16", "shape": [18446744073709551615, 4], "data_offsets": [0, 2]}}"#,
        br#"{"a": {"dtype": "BF16", "shape": [1], "data_offsets": [18446744073709551615, 18446744073709551615]}}"#,
    ] {
        assert!(
            parse_header(FILE, hostile_header, None).is_err(),
            "{} should be refused",
            String::from_utf8_lossy(hostile_header)
        );
    }
}

#[test]
fn should_refuse_offsets_that_would_overflow_even_with_no_known_file_size() {
    let header =
        br#"{"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 18446744073709551615]}}"#;
    assert!(parse_header(FILE, header, None).is_err());
}

#[test]
fn should_refuse_a_tensor_running_past_the_file_and_accept_one_ending_at_it() {
    let header = br#"{"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}"#;
    let exact_size = 8 + header.len() as u64 + 4;
    assert!(parse_header(FILE, header, Some(exact_size)).is_ok());
    assert!(parse_header(FILE, header, Some(exact_size - 1)).is_err());
}

#[test]
fn should_read_a_header_length_and_refuse_absurd_or_short_prefixes() {
    assert_eq!(parse_header_length(FILE, &5_u64.to_le_bytes()).unwrap(), 5);
    assert_eq!(
        parse_header_length(FILE, &MAXIMUM_HEADER_BYTES.to_le_bytes()).unwrap(),
        MAXIMUM_HEADER_BYTES
    );
    assert!(parse_header_length(FILE, &(MAXIMUM_HEADER_BYTES + 1).to_le_bytes()).is_err());
    assert!(parse_header_length(FILE, &[0_u8; 7]).is_err());
}
