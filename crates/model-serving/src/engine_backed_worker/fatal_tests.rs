use astronomical_ipc_protocol::ChatGenerationFailureReason;

use super::public_fatal_failure_reason;

const METAL_OUT_OF_MEMORY: &str = "worker inference engine generation failed: fatal engine error: direct MLX execution failed: failed to asynchronously evaluate an MLX array group: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).";

#[test]
fn should_name_running_out_of_graphics_memory_as_its_own_reason() {
    assert_eq!(
        public_fatal_failure_reason(METAL_OUT_OF_MEMORY),
        ChatGenerationFailureReason::OutOfGraphicsMemory
    );
}

/// Only Metal's own report counts: the sentence without the kernel code comes
/// from other subsystems, and must not tell the person to free memory.
#[test]
fn should_not_blame_memory_for_a_failure_that_only_mentions_it() {
    for unrelated_failure in [
        "Insufficient Memory",
        "[METAL] Command buffer execution failed: Insufficient Memory",
        "kIOGPUCommandBufferCallbackErrorOutOfMemory",
        "",
    ] {
        assert!(
            matches!(
                public_fatal_failure_reason(unrelated_failure),
                ChatGenerationFailureReason::FatalExecution { .. }
            ),
            "{unrelated_failure:?}"
        );
    }
}

#[test]
fn should_keep_the_buffer_limit_advice_and_hide_every_other_native_detail() {
    let ChatGenerationFailureReason::FatalExecution {
        reason: buffer_limit_reason,
    } = public_fatal_failure_reason(
        "[metal::malloc] Attempting to allocate 34359738368 bytes which is greater than the maximum allowed buffer size",
    )
    else {
        panic!("a buffer-limit failure stays a fatal execution reason");
    };
    let ChatGenerationFailureReason::FatalExecution {
        reason: other_reason,
    } = public_fatal_failure_reason("some native error at /a/local/path/transforms.cpp:15")
    else {
        panic!("an unrecognised failure stays a fatal execution reason");
    };

    assert!(buffer_limit_reason.contains("platform buffer limit"));
    assert!(!buffer_limit_reason.contains("34359738368"));
    assert!(other_reason.contains("inspect the worker log"));
    assert!(!other_reason.contains("/a/local/path"));
}

/// The MLX runtime decides whether to recover from this error, and the worker
/// decides how to word it once recovery has failed. Both must recognise the
/// same text, or a failure the runtime treated as memory would be worded as
/// something else.
#[cfg(feature = "direct-mlx")]
#[test]
fn should_agree_with_the_mlx_runtime_about_what_out_of_memory_looks_like() {
    for runtime_description in [
        "[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).",
        "Insufficient Memory",
        "kIOGPUCommandBufferCallbackErrorOutOfMemory",
    ] {
        let runtime_error = astronomical_runtime_integration::MlxRuntimeError::RuntimeOperation {
            operation: "evaluate an MLX array group",
            description: runtime_description.to_owned(),
        };

        assert_eq!(
            super::describes_graphics_processor_out_of_memory(runtime_description),
            runtime_error.is_recoverable_graphics_processor_out_of_memory(),
            "{runtime_description:?}"
        );
    }
}
