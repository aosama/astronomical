import Foundation

/**
 * The failure vocabulary of the MLX runtime I/O surface, continuing the
 * Rust `MlxRuntimeError` contract at the Swift boundary. The nested
 * `gpuDeviceUnavailable` case stays on `MlxRuntime.MlxRuntimeError` in
 * `GpuDeviceInfo.swift`; this top-level enum carries the SafeTensors
 * I/O failures because a nested enum cannot gain cases from another
 * file.
 */
public enum MlxRuntimeError: Error, Equatable {

    /// A tensor lookup by safetensors header name found no entry.
    case tensorLookupFailed(tensorName: String)

    /// A bounded serialization produced more bytes than the caller's
    /// maximum allows.
    case safetensorsSerializationLimitExceeded(attemptedByteCount: Int, maximumByteCount: Int)

    /// A set of bounded read intervals does not tile its payload exactly
    /// or overlaps itself in the source file.
    case boundedIntervalValidation(description: String)

    /// A positional (pread-based) file read failed or hit an unexpected
    /// end of file.
    case positionalReadFailed(description: String)

    /// A runtime operation failed for the stated reason; the operation
    /// name keeps the failure attributable to its owning step.
    case runtimeOperation(operation: String, description: String)
}
