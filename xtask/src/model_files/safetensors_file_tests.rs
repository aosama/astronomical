//! Unit contracts for the safetensors reader and writer: a written file
//! reads back as planned, and malformed files and wrong plans are refused.

use std::io::Write;

use super::{
    OutputCheckpoint, PlannedTensor, SourceCheckpoint, TensorElementType, output_header_bytes,
};

fn planned(tensor_name: &str, element_type: TensorElementType, shape: &[u64]) -> PlannedTensor {
    PlannedTensor {
        tensor_name: tensor_name.to_owned(),
        element_type,
        shape: shape.to_vec(),
    }
}

/// Writes a file from a header and data, returning the folder that owns it.
fn checkpoint_file_with(header_json: &str, tensor_data: &[u8]) -> tempfile::TempDir {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let mut checkpoint_bytes = (header_json.len() as u64).to_le_bytes().to_vec();
    checkpoint_bytes.extend_from_slice(header_json.as_bytes());
    checkpoint_bytes.extend_from_slice(tensor_data);
    std::fs::write(
        temporary_directory.path().join("model.safetensors"),
        checkpoint_bytes,
    )
    .expect("the checkpoint file should be writable");
    temporary_directory
}

#[test]
fn should_read_back_the_tensors_it_wrote_in_name_order() {
    let planned_tensors = vec![
        planned("zeta.weight", TensorElementType::Float16, &[2, 2]),
        planned("alpha.weight", TensorElementType::Unsigned32, &[1, 2]),
    ];
    let mut written_bytes = Vec::new();
    let mut output_checkpoint =
        OutputCheckpoint::begin(&mut written_bytes, planned_tensors).expect("the header writes");
    output_checkpoint
        .append_tensor("zeta.weight", &[1, 2, 3, 4, 5, 6, 7, 8])
        .expect("the first tensor matches its plan");
    output_checkpoint
        .append_tensor("alpha.weight", &[9, 10, 11, 12, 13, 14, 15, 16])
        .expect("the second tensor matches its plan");
    output_checkpoint
        .finish()
        .expect("every tensor was written");
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let weights_path = temporary_directory.path().join("model.safetensors");
    std::fs::File::create(&weights_path)
        .and_then(|mut weights_file| weights_file.write_all(&written_bytes))
        .expect("the written bytes should be saved");

    let mut source_checkpoint =
        SourceCheckpoint::open(&weights_path).expect("the written file should read back");

    let tensor_records = source_checkpoint.tensor_records.clone();
    assert_eq!(tensor_records[0].tensor_name, "alpha.weight");
    assert_eq!(
        tensor_records[0].element_type,
        TensorElementType::Unsigned32
    );
    assert_eq!(tensor_records[1].shape, vec![2, 2]);
    assert_eq!(
        source_checkpoint
            .read_tensor_bytes(&tensor_records[0])
            .expect("the tensor should be readable"),
        vec![9, 10, 11, 12, 13, 14, 15, 16]
    );
    assert_eq!(
        source_checkpoint
            .read_tensor_bytes(&tensor_records[1])
            .expect("the tensor should be readable"),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[test]
fn should_pad_the_header_so_tensor_data_starts_on_an_eight_byte_boundary() {
    for tensor_name in [
        "a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "abcdefgh",
    ] {
        let header_bytes =
            output_header_bytes(&[planned(tensor_name, TensorElementType::Float16, &[1])]);

        assert_eq!(header_bytes.len() % 8, 0);
        let declared_length = u64::from_le_bytes(
            header_bytes[..8]
                .try_into()
                .expect("the length prefix is eight bytes"),
        );
        assert_eq!(declared_length as usize, header_bytes.len() - 8);
    }
}

#[test]
fn should_mark_the_output_as_already_in_the_mlx_layout() {
    let header_bytes = output_header_bytes(&[planned("a", TensorElementType::Float16, &[1])]);

    let header: serde_json::Value =
        serde_json::from_slice(&header_bytes[8..]).expect("the header should be JSON");

    assert_eq!(header["__metadata__"]["format"], "mlx");
    assert_eq!(header["a"]["data_offsets"], serde_json::json!([0, 2]));
}

#[test]
fn should_refuse_a_tensor_that_does_not_match_its_plan() {
    let mut written_bytes = Vec::new();
    let mut output_checkpoint = OutputCheckpoint::begin(
        &mut written_bytes,
        vec![planned("a.weight", TensorElementType::Float16, &[2])],
    )
    .expect("the header writes");

    assert!(
        output_checkpoint
            .append_tensor("a.weight", &[1, 2])
            .is_err(),
        "a tensor shorter than planned must be refused"
    );
    assert!(
        output_checkpoint
            .append_tensor("b.weight", &[1, 2, 3, 4])
            .is_err(),
        "a tensor under another name must be refused"
    );
}

#[test]
fn should_refuse_to_finish_while_a_planned_tensor_is_missing() {
    let mut written_bytes = Vec::new();
    let output_checkpoint = OutputCheckpoint::begin(
        &mut written_bytes,
        vec![planned("a.weight", TensorElementType::Float16, &[2])],
    )
    .expect("the header writes");

    assert!(output_checkpoint.finish().is_err());
}

#[test]
fn should_refuse_a_tensor_whose_byte_range_disagrees_with_its_shape() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"a":{"dtype":"F16","shape":[2,2],"data_offsets":[0,6]}}"#,
        &[0; 6],
    );

    let open_outcome =
        SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors"));

    assert!(open_outcome.is_err());
}

#[test]
fn should_refuse_an_element_type_no_converter_reads() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
        &[0; 8],
    );

    let open_outcome =
        SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors"));

    assert!(matches!(open_outcome, Err(reason) if reason.contains("I64")));
}

#[test]
fn should_refuse_a_header_length_beyond_the_limit_without_allocating_it() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let weights_path = temporary_directory.path().join("model.safetensors");
    std::fs::write(&weights_path, u64::MAX.to_le_bytes())
        .expect("the corrupt file should be writable");

    assert!(SourceCheckpoint::open(&weights_path).is_err());
}

#[test]
fn should_ignore_the_metadata_entry_of_a_source_header() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );

    let source_checkpoint =
        SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors"))
            .expect("a header with metadata should open");

    assert_eq!(source_checkpoint.tensor_records.len(), 1);
}

#[test]
fn should_refuse_a_tensor_that_ends_past_the_end_of_the_file() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
        &[0; 6],
    );

    let open_outcome =
        SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors"));

    assert!(matches!(open_outcome, Err(reason) if reason.contains("past the 6 bytes")));
}

#[test]
fn should_accept_a_tensor_that_ends_exactly_at_the_end_of_the_file() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
        &[0; 8],
    );

    assert!(SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors")).is_ok());
}

#[test]
fn should_read_a_bfloat16_tensor_record() {
    let checkpoint_directory = checkpoint_file_with(
        r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#,
        &[0; 12],
    );

    let source_checkpoint =
        SourceCheckpoint::open(&checkpoint_directory.path().join("model.safetensors"))
            .expect("a bfloat16 tensor should open");

    assert_eq!(
        source_checkpoint.tensor_records[0].element_type,
        TensorElementType::BFloat16
    );
    assert_eq!(source_checkpoint.tensor_records[0].byte_count, 12);
}
