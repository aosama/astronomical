import Foundation

import MLX

/** What a completed SafeTensors write left on disk: the written byte count. */
public struct SafetensorsWriteOutcome: Equatable {

    /// The file byte size measured after the write completed.
    public let writtenByteCount: Int

    public init(writtenByteCount: Int) {
        self.writtenByteCount = writtenByteCount
    }
}

/**
 * The failure vocabulary of the SafeTensors writer surface. Descriptor
 * problems are separated from MLX's native serialization errors so the
 * write journeys can assert the exact failure class.
 */
public enum SafetensorsWriterError: Error, Equatable {

    /// The destination descriptor could not be opened, closed, or measured.
    case descriptorIo(description: String)

    /// A save was attempted with an empty array dictionary.
    case atLeastOneArrayRequired
}

/**
 * Writes named arrays to a SafeTensors file through MLX's stock saver,
 * continuing the Rust `SafetensorsFileWriter` contract.
 *
 * Divergence from Rust, by design: MLX's saver writes to a URL, not an
 * open FD (file descriptor), so the descriptor-I/O failure class is probed
 * by opening the destination for writing first; that probe is what
 * surfaces unwritable destinations (for example a read-only file) before
 * any bytes are produced.
 */
enum SafetensorsFileWriter {

    /**
     * - Parameters:
     *   - arrays: the named arrays to serialize; must not be empty.
     *   - metadata: the header metadata map to embed.
     *   - destinationUrl: the `.safetensors` file URL to write.
     * - Returns: the measured on-disk byte count of the completed write.
     * - Throws: `SafetensorsWriterError.atLeastOneArrayRequired` for an
     *   empty array dictionary; `SafetensorsWriterError.descriptorIo` when
     *   the destination cannot be opened, closed, or measured; MLX's
     *   native serialization error passes through unchanged.
     */
    static func save(
        arrays: [String: MLXArray],
        metadata: [String: String],
        to destinationUrl: URL
    ) throws -> SafetensorsWriteOutcome {
        guard (!arrays.isEmpty) else {
            throw SafetensorsWriterError.atLeastOneArrayRequired
        }
        let writeProbeHandle: FileHandle
        do {
            writeProbeHandle = try FileHandle(forWritingTo: destinationUrl)
        } catch {
            throw SafetensorsWriterError.descriptorIo(
                description: "the destination could not be opened for writing: \(destinationUrl.lastPathComponent)")
        }
        do {
            try writeProbeHandle.close()
        } catch {
            throw SafetensorsWriterError.descriptorIo(
                description: "the destination could not be closed after the write probe: \(destinationUrl.lastPathComponent)")
        }

        try MLX.save(arrays: arrays, metadata: metadata, url: destinationUrl)

        let destinationAttributes: [FileAttributeKey: Any]
        do {
            destinationAttributes = try FileManager.default.attributesOfItem(atPath: destinationUrl.path)
        } catch {
            throw SafetensorsWriterError.descriptorIo(
                description: "the written file's attributes could not be read: \(destinationUrl.lastPathComponent)")
        }
        guard let writtenByteCount: Int = destinationAttributes[.size] as? Int else {
            throw SafetensorsWriterError.descriptorIo(
                description: "the written file's byte size could not be determined: \(destinationUrl.lastPathComponent)")
        }
        return SafetensorsWriteOutcome(writtenByteCount: writtenByteCount)
    }
}
