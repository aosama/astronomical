use std::fs::{File, OpenOptions};
use std::io::Write;

use astronomical_runtime_integration::MlxDtype;

#[test]
fn should_save_multiple_arrays_and_metadata_through_a_retained_file_descriptor() {
    let runtime = crate::common::runtime_test_support::runtime();
    let first_tensor = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4])
        .expect("the test should create the first tensor");
    let second_tensor = runtime
        .zeros(&[2, 3], MlxDtype::BFloat16)
        .expect("the test should create the second tensor");
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let cache_file_path = temporary_directory.path().join("cache-block.safetensors");
    let cache_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&cache_file_path)
        .expect("the test should create a new cache file");

    runtime
        .save_safetensors(
            cache_file,
            &[
                ("layer_0_recurrent", &first_tensor),
                ("layer_1_keys", &second_tensor),
            ],
            &[("format_version", "1"), ("token_count", "2048")],
        )
        .expect("MLX should save tensors through the retained descriptor");

    let saved_cache_file =
        File::open(&cache_file_path).expect("the test should reopen the saved cache file");
    let saved_tensors = runtime
        .load_safetensors(saved_cache_file, None)
        .expect("MLX should load the file written through its official writer");
    let restored_first_tensor = saved_tensors
        .tensor("layer_0_recurrent")
        .expect("the first tensor should round trip");
    let restored_second_tensor = saved_tensors
        .tensor("layer_1_keys")
        .expect("the second tensor should round trip");

    assert_eq!(
        restored_first_tensor
            .to_vec_f32()
            .expect("the first tensor should evaluate"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(restored_second_tensor.shape(), vec![2, 3]);
    assert_eq!(restored_second_tensor.dtype(), MlxDtype::BFloat16);
}

#[test]
fn should_serialize_safetensors_to_memory_before_background_file_writing() {
    let runtime = crate::common::runtime_test_support::runtime();
    let source_tensor = runtime
        .array_from_f32(&[2.0, 4.0, 6.0, 8.0], &[2, 2])
        .expect("the test should create the source tensor");

    let serialized_safetensors_bytes = runtime
        .serialize_safetensors(
            &[("layer_0_keys", &source_tensor)],
            &[("format_version", "test")],
        )
        .expect("MLX should serialize safetensors without filesystem access");

    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let serialized_file_path = temporary_directory.path().join("serialized.safetensors");
    let mut serialized_file =
        File::create(&serialized_file_path).expect("the test should create the serialized file");
    serialized_file
        .write_all(&serialized_safetensors_bytes)
        .expect("the test should persist serialized bytes");
    drop(serialized_file);

    let restored_safetensors = runtime
        .load_safetensors(
            File::open(serialized_file_path).expect("the test should reopen serialized bytes"),
            None,
        )
        .expect("MLX should load the memory-serialized safetensors");
    assert_eq!(
        restored_safetensors
            .tensor("layer_0_keys")
            .expect("the serialized tensor should exist")
            .to_vec_f32()
            .expect("the serialized tensor should evaluate"),
        vec![2.0, 4.0, 6.0, 8.0]
    );
}
