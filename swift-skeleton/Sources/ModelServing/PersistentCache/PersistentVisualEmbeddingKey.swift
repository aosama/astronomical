import Foundation;

import CryptoKit;

/// Stable identity for one exact encoded image's projected visual
/// embeddings, port of the Rust `PersistentVisualEmbeddingKey`. The model
/// ID and revision bind this identity to the validated model namespace so
/// a different model revision automatically produces a different hash
/// without any code change.
public struct PersistentVisualEmbeddingKey: Equatable, Sendable {

    /// The persisted projected visual-embedding tensor contract version.
    public static let FORMAT_VERSION: String = "2";

    static let HASH_DOMAIN: [UInt8] = Array(
        "astronomical-qwen3-5-moe-visual-embedding".utf8);

    public let visualEmbeddingHash: Data;

    public let encodedImageSha256: Data;

    /// Creates the model- and format-isolated identity for one encoded
    /// image.
    public static func forImage(
        encodedImageSha256: Data,
        modelId: String,
        modelRevision: String
    ) -> PersistentVisualEmbeddingKey {
        var contentHasher: SHA256 = SHA256();
        Self.updateLengthPrefixedBytes(
            contentHasher: &contentHasher, byteSequence: Self.HASH_DOMAIN);
        Self.updateLengthPrefixedBytes(
            contentHasher: &contentHasher,
            byteSequence: Array(PersistentVisualEmbeddingKey.FORMAT_VERSION.utf8));
        Self.updateLengthPrefixedBytes(
            contentHasher: &contentHasher, byteSequence: Array(modelId.utf8));
        Self.updateLengthPrefixedBytes(
            contentHasher: &contentHasher, byteSequence: Array(modelRevision.utf8));
        Self.updateLengthPrefixedBytes(
            contentHasher: &contentHasher, byteSequence: [UInt8](encodedImageSha256));
        return PersistentVisualEmbeddingKey(
            visualEmbeddingHash: Data(contentHasher.finalize()),
            encodedImageSha256: encodedImageSha256);
    }

    /// Every hashed field is prefixed with its big-endian byte length so
    /// distinct field boundaries can never concatenate into the same digest.
    private static func updateLengthPrefixedBytes(
        contentHasher: inout SHA256,
        byteSequence: [UInt8]
    ) {
        var bigEndianLength: UInt64 = UInt64(byteSequence.count).bigEndian;
        withUnsafeBytes(of: &bigEndianLength) { (valueBuffer: UnsafeRawBufferPointer) in
            contentHasher.update(data: Data(valueBuffer));
        };
        byteSequence.withUnsafeBytes { (valueBuffer: UnsafeRawBufferPointer) in
            contentHasher.update(data: Data(valueBuffer));
        };
    }
}
