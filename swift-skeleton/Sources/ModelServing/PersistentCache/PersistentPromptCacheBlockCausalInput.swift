import Foundation;

import CryptoKit;

/// Canonical non-token causal input introduced by one prompt-cache block,
/// port of the Rust `PersistentPromptCacheBlockCausalInput`. Model families
/// own the canonical bytes because only they understand which non-token
/// inputs enter the decoder at each prompt position; the persistent cache
/// treats those bytes as opaque identity and never interprets model syntax.
public struct PersistentPromptCacheBlockCausalInput: Equatable, Sendable {

    private let canonicalDigestData: Data?;

    /// Represents a block whose decoder inputs are fully identified by its
    /// tokens and ancestry.
    public init() {
        self.canonicalDigestData = nil;
    }

    /// Reduces model-owned canonical identity to one bounded digest.
    public init(canonicalBytes: Data) {
        if canonicalBytes.isEmpty {
            self.canonicalDigestData = nil;
            return;
        }
        let digest: SHA256Digest = SHA256.hash(data: canonicalBytes);
        self.canonicalDigestData = Data(digest);
    }

    private init(canonicalDigestData: Data?) {
        self.canonicalDigestData = canonicalDigestData;
    }

    /// The empty causal input: no additional non-token identity.
    public static func empty() -> PersistentPromptCacheBlockCausalInput {
        return PersistentPromptCacheBlockCausalInput(canonicalDigestData: nil);
    }

    /// Returns whether this block introduces no additional non-token input.
    public func isEmpty() -> Bool {
        return self.canonicalDigestData == nil;
    }

    /// The bounded digest the block-key chain folds in, when present.
    internal func canonicalDigest() -> Data? {
        return self.canonicalDigestData;
    }
}
