import Foundation

import MLX

/**
 * Loads a whole SafeTensors weights file into one retained buffer,
 * continuing the Rust `RetainedFileSafetensorsLoader` contract: read every
 * byte of an already-open descriptor positionally, then decode with MLX's
 * stock in-memory loader.
 *
 * Divergence from Rust, by design: MLX-Swift has no mmap-backed lazy file
 * reader, so the read is eager and the assembled bytes are retained on the
 * returned `SafetensorsFile`.
 */
enum RetainedFileSafetensorsLoader {

    /**
     * - Parameters:
     *   - weightsFile: the open weights file descriptor to read from.
     *   - positionalFileReadMetrics: optional instrumentation; the read is
     *     measured for overlap, latency, and volume when attached.
     * - Returns: the decoded tensors plus the retained assembled bytes.
     * - Throws: `MlxRuntimeError.runtimeOperation` when the file is empty;
     *   `MlxRuntimeError.positionalReadFailed` when the read fails; and
     *   MLX's native decode error for malformed SafeTensors bytes.
     */
    static func load(
        weightsFile: FileHandle,
        positionalFileReadMetrics: PositionalFileReadMetrics?
    ) throws -> SafetensorsFile {
        let weightsFileByteCount: Int = try PositionalFileReader.fileByteCount(
            fileDescriptor: weightsFile.fileDescriptor)
        guard weightsFileByteCount > 0 else {
            throw MlxRuntimeError.runtimeOperation(
                operation: "load safetensors",
                description: "the retained weights file is empty")
        }
        let assembledBytes: Data = try PositionalFileReader.read(
            fileDescriptor: weightsFile.fileDescriptor,
            headerBytes: Data(),
            payloadRequests: [PositionalFileReader.ReadRequest(
                sourceFileOffset: 0,
                destinationBufferOffset: 0,
                byteCount: weightsFileByteCount)],
            maximumConcurrentReadCount: 1,
            metrics: positionalFileReadMetrics)
        let tensorsByName: [String: MLXArray] = try MLX.loadArrays(data: assembledBytes)
        return SafetensorsFile(tensorsByName: tensorsByName, assembledBytes: assembledBytes)
    }
}
