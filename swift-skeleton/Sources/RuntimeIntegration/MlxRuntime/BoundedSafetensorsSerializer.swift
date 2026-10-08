import Foundation

import MLX

/**
 * In-memory SafeTensors serialization through MLX's stock serializer,
 * continuing the Rust `BoundedSafetensorsSerializer` contract: an
 * unbounded form, and a bounded form that refuses byte counts above the
 * caller's ceiling so oversized payloads fail before their bytes are
 * ever paged anywhere.
 */
enum BoundedSafetensorsSerializer {

    /**
     * - Returns: the serialized SafeTensors bytes of `arrays` with
     *   `metadata` embedded.
     * - Throws: MLX's native serialization error for empty or invalid
     *   input; nothing is wrapped or translated.
     */
    static func serialize(arrays: [String: MLXArray], metadata: [String: String]) throws -> Data {
        return try MLX.saveToData(arrays: arrays, metadata: metadata)
    }

    /**
     * - Parameters:
     *   - arrays: the named arrays to serialize; must not be empty.
     *   - metadata: the header metadata map to embed.
     *   - maximumByteCount: the serialization ceiling in bytes.
     * - Returns: the serialized bytes when they fit the ceiling.
     * - Throws: `MlxRuntimeError.runtimeOperation` for an empty array
     *   dictionary; `MlxRuntimeError.safetensorsSerializationLimitExceeded`
     *   when the serialized count exceeds `maximumByteCount`; MLX's native
     *   serialization error passes through unchanged.
     */
    static func serialize(
        arrays: [String: MLXArray],
        metadata: [String: String],
        maximumByteCount: Int
    ) throws -> Data {
        guard (!arrays.isEmpty) else {
            throw MlxRuntimeError.runtimeOperation(
                operation: "serialize safetensors",
                description: "at least one named array is required")
        }
        let serializedBytes: Data = try MLX.saveToData(arrays: arrays, metadata: metadata)
        if (serializedBytes.count > maximumByteCount) {
            throw MlxRuntimeError.safetensorsSerializationLimitExceeded(
                attemptedByteCount: serializedBytes.count,
                maximumByteCount: maximumByteCount)
        }
        return serializedBytes
    }
}
