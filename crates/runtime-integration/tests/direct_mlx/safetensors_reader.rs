use std::{
    fs::{self, File},
    sync::Arc,
    time::Instant,
};

use astronomical_runtime_integration::{
    BoundedReadInterval, MlxDtype, MlxMemoryLimits, MlxRuntime, MlxRuntimeError,
    PositionalFileReadMetrics,
};

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const CONCURRENT_READ_TENSOR_BYTE_COUNT: usize = 16 * 1024 * 1024;

#[test]
fn should_async_evaluate_safetensors_after_dropping_the_load_result_and_removing_its_path() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let weights_path = temporary_directory.path().join("model.safetensors");
    fs::write(&weights_path, tiny_safetensors_bytes())
        .expect("the test should write a tiny safetensors file");
    let retained_weights_file =
        File::open(&weights_path).expect("the test should retain a read-only weights handle");
    fs::remove_file(&weights_path).expect("the test should remove the mutable path identity");

    let memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    let runtime =
        MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize");
    let weights = runtime
        .load_safetensors(retained_weights_file, None)
        .expect("MLX should load through the retained file descriptor");
    assert!(matches!(
        weights.tensor("missing.weight"),
        Err(MlxRuntimeError::RuntimeOperation {
            operation: "look up a safetensors tensor",
            ..
        })
    ));
    let embedding_weight = weights
        .tensor("model.embed_tokens.weight")
        .expect("the checked tensor should be present");
    runtime
        .async_eval_arrays(&[&embedding_weight])
        .expect("the lazy safetensors tensor should submit for asynchronous evaluation");
    drop(weights);

    assert_eq!(embedding_weight.shape(), vec![1, 4]);
    assert_eq!(embedding_weight.dtype(), MlxDtype::Float32);
    assert_eq!(embedding_weight.element_count(), 4);
    assert_eq!(embedding_weight.byte_count(), 16);
    embedding_weight
        .evaluate()
        .expect("lazy tensor evaluation should still use the retained descriptor");
    assert_eq!(
        embedding_weight
            .to_vec_f32()
            .expect("the evaluated safetensors values should copy to the host"),
        vec![1.0, 2.0, 3.0, 4.0],
    );
}

#[test]
fn should_async_evaluate_multiple_tensors_from_independent_bounded_source_intervals() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let weights_path = temporary_directory.path().join("bounded-source.bin");
    let mut source_bytes = vec![0_u8; 40];
    source_bytes[4..12].copy_from_slice(&[1.0_f32, 2.0_f32].map(f32::to_le_bytes).concat());
    source_bytes[28..36].copy_from_slice(&[3.0_f32, 4.0_f32].map(f32::to_le_bytes).concat());
    fs::write(&weights_path, source_bytes)
        .expect("the test should write separated tensor payloads");

    let memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    let runtime =
        MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize");
    let expert_file_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let bounded_load_result = runtime
        .load_safetensors_from_bounded_ranges(
            File::open(&weights_path).expect("the test should open its bounded source"),
            two_tensor_synthetic_safetensors_header(),
            vec![
                BoundedReadInterval {
                    virtual_payload_offset: 0,
                    source_file_offset: 4,
                    source_byte_count: 8,
                },
                BoundedReadInterval {
                    virtual_payload_offset: 8,
                    source_file_offset: 28,
                    source_byte_count: 8,
                },
            ],
            16,
            Some(Arc::clone(&expert_file_read_metrics)),
        )
        .expect("the bounded multi-range reader should construct lazy tensors");
    let first_weight = bounded_load_result
        .tensor("first.weight")
        .expect("the first bounded tensor should exist");
    let second_weight = bounded_load_result
        .tensor("second.weight")
        .expect("the second bounded tensor should exist");

    runtime
        .async_eval_arrays(&[&first_weight, &second_weight])
        .expect("both bounded tensors should submit together");
    drop(bounded_load_result);

    assert_eq!(
        first_weight
            .to_vec_f32()
            .expect("the first bounded tensor should evaluate"),
        vec![1.0, 2.0],
    );
    assert_eq!(
        second_weight
            .to_vec_f32()
            .expect("the second bounded tensor should evaluate"),
        vec![3.0, 4.0],
    );
    let expert_file_read_snapshot = expert_file_read_metrics.snapshot();
    assert_eq!(expert_file_read_snapshot.read_call_count, 2);
    assert_eq!(expert_file_read_snapshot.read_byte_count, 16);
    assert!(expert_file_read_snapshot.total_read_elapsed_nanoseconds > 0);
    assert!(expert_file_read_snapshot.maximum_read_elapsed_nanoseconds > 0);
    assert_eq!(expert_file_read_snapshot.read_failure_count, 0);
}

#[test]
#[ignore = "measures positional file-read concurrency through one MLX safetensors reader"]
fn should_measure_positional_reads_for_multiple_tensors_from_one_safetensors_reader() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let weights_path = temporary_directory
        .path()
        .join("concurrent-read.safetensors");
    fs::write(
        &weights_path,
        four_tensor_safetensors_bytes(CONCURRENT_READ_TENSOR_BYTE_COUNT),
    )
    .expect("the test should write four substantial safetensors payloads");

    let memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    let runtime =
        MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize");
    let positional_file_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let weights = runtime
        .load_safetensors(
            File::open(&weights_path).expect("the test should open its safetensors file"),
            Some(Arc::clone(&positional_file_read_metrics)),
        )
        .expect("MLX should construct four lazy tensors from one reader");
    let first_weight = weights
        .tensor("first.weight")
        .expect("the first tensor should exist");
    let second_weight = weights
        .tensor("second.weight")
        .expect("the second tensor should exist");
    let third_weight = weights
        .tensor("third.weight")
        .expect("the third tensor should exist");
    let fourth_weight = weights
        .tensor("fourth.weight")
        .expect("the fourth tensor should exist");

    let concurrent_evaluation_started_at = Instant::now();
    runtime
        .evaluate_arrays(&[&first_weight, &second_weight, &third_weight, &fourth_weight])
        .expect("MLX should evaluate all four lazy tensors together");
    let concurrent_evaluation_elapsed = concurrent_evaluation_started_at.elapsed();

    let positional_file_read_snapshot = positional_file_read_metrics.snapshot();
    assert_eq!(positional_file_read_snapshot.read_call_count, 4);
    assert_eq!(
        positional_file_read_snapshot.read_byte_count,
        u64::try_from(CONCURRENT_READ_TENSOR_BYTE_COUNT * 4)
            .expect("the test payload byte count should fit u64")
    );
    eprintln!(
        "[safetensors-positional-read] status=success tensors=4 bytes={} maximum_concurrent_reads={} summed_read_milliseconds={:.3} evaluation_milliseconds={:.3}",
        positional_file_read_snapshot.read_byte_count,
        positional_file_read_snapshot.maximum_concurrent_read_count,
        positional_file_read_snapshot.total_read_elapsed_nanoseconds as f64 / 1_000_000.0,
        concurrent_evaluation_elapsed.as_secs_f64() * 1_000.0,
    );
}

fn tiny_safetensors_bytes() -> Vec<u8> {
    let tensor_payload_bytes = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<u8>>();
    let header = format!(
        r#"{{"model.embed_tokens.weight":{{"dtype":"F32","shape":[1,4],"data_offsets":[0,{}]}}}}"#,
        tensor_payload_bytes.len()
    );
    let mut safetensors_bytes = Vec::new();
    safetensors_bytes.extend_from_slice(
        &u64::try_from(header.len())
            .expect("the test header length should fit u64")
            .to_le_bytes(),
    );
    safetensors_bytes.extend_from_slice(header.as_bytes());
    safetensors_bytes.extend_from_slice(&tensor_payload_bytes);
    safetensors_bytes
}

fn two_tensor_synthetic_safetensors_header() -> Vec<u8> {
    let encoded_header = br#"{"first.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"second.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    let mut synthetic_header_bytes = Vec::with_capacity(8 + encoded_header.len());
    synthetic_header_bytes.extend_from_slice(
        &u64::try_from(encoded_header.len())
            .expect("the synthetic header length should fit u64")
            .to_le_bytes(),
    );
    synthetic_header_bytes.extend_from_slice(encoded_header);
    synthetic_header_bytes
}

fn four_tensor_safetensors_bytes(tensor_byte_count: usize) -> Vec<u8> {
    assert!(tensor_byte_count.is_multiple_of(size_of::<f32>()));
    let tensor_element_count = tensor_byte_count / size_of::<f32>();
    let header = format!(
        r#"{{"first.weight":{{"dtype":"F32","shape":[{tensor_element_count}],"data_offsets":[0,{tensor_byte_count}]}},"second.weight":{{"dtype":"F32","shape":[{tensor_element_count}],"data_offsets":[{tensor_byte_count},{}]}},"third.weight":{{"dtype":"F32","shape":[{tensor_element_count}],"data_offsets":[{},{}]}},"fourth.weight":{{"dtype":"F32","shape":[{tensor_element_count}],"data_offsets":[{},{}]}}}}"#,
        tensor_byte_count * 2,
        tensor_byte_count * 2,
        tensor_byte_count * 3,
        tensor_byte_count * 3,
        tensor_byte_count * 4,
    );
    let total_payload_byte_count = tensor_byte_count * 4;
    let mut safetensors_bytes = Vec::with_capacity(8 + header.len() + total_payload_byte_count);
    safetensors_bytes.extend_from_slice(
        &u64::try_from(header.len())
            .expect("the concurrent-read header length should fit u64")
            .to_le_bytes(),
    );
    safetensors_bytes.extend_from_slice(header.as_bytes());
    safetensors_bytes.resize(8 + header.len() + total_payload_byte_count, 0);
    safetensors_bytes
}
