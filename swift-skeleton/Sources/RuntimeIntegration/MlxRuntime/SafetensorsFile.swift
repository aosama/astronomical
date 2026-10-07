import Foundation

import MLX

/**
 * The result of any SafeTensors load through this module: the decoded
 * tensors plus the assembled bytes they were decoded from.
 *
 * `assembledBytes` is retained deliberately. MLX-Swift has no mmap-backed
 * lazy file loading, so loads here are eager positional reads into one
 * buffer; keeping the buffer alive guarantees lazily evaluated arrays
 * always have addressable storage. This is the documented divergence from
 * the Rust FD (file descriptor) + mmap reader, which could drop the
 * mapping after decode.
 */
public struct SafetensorsFile {

    /// Every tensor the SafeTensors header named, keyed by header name.
    public let tensorsByName: [String: MLXArray]

    /// The header-plus-payload bytes the tensors were decoded from.
    /// Retained: lazy arrays may reference this storage.
    public let assembledBytes: Data

    public init(tensorsByName: [String: MLXArray], assembledBytes: Data) {
        self.tensorsByName = tensorsByName
        self.assembledBytes = assembledBytes
    }

    /**
     * - Parameters:
     *   - tensorName: the SafeTensors header name to look up.
     * - Returns: the tensor stored under `tensorName`.
     * - Throws: `MlxRuntimeError.tensorLookupFailed` when no tensor is
     *   stored under that name.
     */
    public func tensor(_ tensorName: String) throws -> MLXArray {
        guard let tensor: MLXArray = tensorsByName[tensorName] else {
            throw MlxRuntimeError.tensorLookupFailed(tensorName: tensorName)
        }
        return tensor
    }
}
