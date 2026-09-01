use std::{fs, fs::OpenOptions, os::unix::fs::FileExt};

use astronomical_runtime_integration::{
    MlxMemoryLimits, MlxRuntime, MlxRuntimeError, compiled_metallib_path, validate_metallib_path,
};

const ACTIVE_MEMORY_LIMIT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;

#[test]
fn should_install_the_runtime_and_expose_enforced_memory_limits() {
    let memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("positive ordered memory limits should be valid");

    let mut runtime = MlxRuntime::initialize(memory_limits)
        .expect("the pinned MLX runtime should initialize without terminating the process");

    assert_eq!(runtime.version(), "0.32.2");
    assert_eq!(
        runtime
            .configured_memory_limit_bytes()
            .expect("the active memory limit should be readable"),
        ACTIVE_MEMORY_LIMIT_BYTES
    );
    assert_eq!(runtime.memory_limits(), memory_limits);
    let memory_snapshot = runtime
        .memory_snapshot()
        .expect("runtime memory metrics should be readable");
    assert!(memory_snapshot.active_memory_bytes() <= ACTIVE_MEMORY_LIMIT_BYTES);

    let lowered_memory_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES - 1,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES - 1,
    )
    .expect("the lowered memory limits should be valid");
    runtime
        .update_memory_limits(lowered_memory_limits)
        .expect("the runtime should lower both MLX memory limits");
    assert_eq!(runtime.memory_limits(), lowered_memory_limits);
    assert_eq!(
        runtime
            .configured_memory_limit_bytes()
            .expect("the lowered MLX memory limit should be readable"),
        lowered_memory_limits.active_memory_limit_bytes()
    );

    runtime
        .update_memory_limits(memory_limits)
        .expect("the runtime should raise all MLX memory limits");
    assert_eq!(runtime.memory_limits(), memory_limits);

    let conflicting_limits = MlxMemoryLimits::new(
        ACTIVE_MEMORY_LIMIT_BYTES - 1,
        ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the conflicting limits should be independently valid");
    assert!(matches!(
        MlxRuntime::initialize(conflicting_limits),
        Err(MlxRuntimeError::RuntimeAlreadyConfigured {
            active_memory_limit_bytes: ACTIVE_MEMORY_LIMIT_BYTES,
            allocator_cache_memory_limit_bytes: ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        })
    ));
}

#[test]
fn should_reject_a_zero_active_memory_limit() {
    assert!(matches!(
        MlxMemoryLimits::new(0, ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES),
        Err(MlxRuntimeError::InvalidMemoryLimits {
            description: "active memory limit must be positive"
        })
    ));
}

#[test]
fn should_allow_a_zero_allocator_cache_limit_to_disable_allocator_caching() {
    let memory_limits = MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, 0)
        .expect("MLX defines zero as disabling its allocator cache");

    assert_eq!(
        memory_limits.active_memory_limit_bytes(),
        ACTIVE_MEMORY_LIMIT_BYTES
    );
    assert_eq!(memory_limits.allocator_cache_memory_limit_bytes(), 0);
}

#[test]
fn should_reject_an_allocator_cache_limit_above_the_active_memory_limit() {
    assert!(matches!(
        MlxMemoryLimits::new(ACTIVE_MEMORY_LIMIT_BYTES, ACTIVE_MEMORY_LIMIT_BYTES + 1),
        Err(MlxRuntimeError::InvalidMemoryLimits {
            description: "allocator cache memory limit cannot exceed the active memory limit"
        })
    ));
}

#[test]
fn should_classify_prefixed_native_operation_capacity_errors() {
    let prefixed_native_capacity_error = astronomical_runtime_integration::classify_mlx_error(
        "prepare native expert route",
        "native MLX operation failed: ASTRONOMICAL_MLX_ACTIVE_MEMORY_LIMIT_EXCEEDED active_bytes=35346771214 allocation_bytes=3538944 allowed_bytes=35350000000 at native-operation.cpp:25".to_owned(),
    );

    assert!(matches!(
        prefixed_native_capacity_error,
        MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes: 35_346_771_214,
            attempted_allocation_bytes: 3_538_944,
            allowed_active_memory_bytes: 35_350_000_000,
        }
    ));
}

#[test]
fn should_reject_a_relocated_metallib_whose_build_produced_bytes_were_changed() {
    let relocation_directory =
        tempfile::tempdir().expect("the test should create a relocation directory");
    let relocated_metallib_path = relocation_directory.path().join("mlx.metallib");
    fs::copy(compiled_metallib_path(), &relocated_metallib_path)
        .expect("the test should copy the build-produced metallib");
    validate_metallib_path(&relocated_metallib_path)
        .expect("an exact relocated metallib copy should retain its build-produced identity");

    let relocated_metallib = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&relocated_metallib_path)
        .expect("the test should open its private metallib copy");
    let mut first_byte = [0_u8; 1];
    relocated_metallib
        .read_exact_at(&mut first_byte, 0)
        .expect("the copied metallib should contain at least one byte");
    first_byte[0] ^= 0xff;
    relocated_metallib
        .write_all_at(&first_byte, 0)
        .expect("the test should alter one byte without changing file size");

    assert!(matches!(
        validate_metallib_path(&relocated_metallib_path),
        Err(MlxRuntimeError::InvalidMetallibPath { .. })
    ));
}
