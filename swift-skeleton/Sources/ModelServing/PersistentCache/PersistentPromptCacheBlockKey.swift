import Foundation;

import CryptoKit;

/// Stable, model-isolated hashing for persistent decoder-state blocks, port
/// of the Rust `PersistentPromptCacheBlockKey`. One immutable,
/// content-addressed block identity inside a model-state chain: the digest
/// folds the parent hash, the storage-contract fingerprint, the model-owned
/// causal input, and every block token, so a shared suffix can never restore
/// state produced by a divergent prompt prefix.
public struct PersistentPromptCacheBlockKey: Equatable, Sendable {

    private static let PERSISTENT_PROMPT_CACHE_ROOT_SEED: Data =
        Data("astronomical-decoder-cache-root".utf8);
    private static let PERSISTENT_PROMPT_CACHE_BLOCK_CAUSAL_INPUT_DOMAIN: Data =
        Data("astronomical-decoder-cache-block-causal-input".utf8);

    private let blockHashData: Data;
    private let blockIndexValue: UInt32;
    private let tokenCountValue: UInt32;
    private let blockTokenCountValue: Int;
    private let storageContractFingerprintData: Data;

    /// Hashes the first block under one exact model storage contract.
    public static func forRootBlock(
        modelContract: PersistentPromptCacheModelContract,
        blockTokens: [UInt32]
    ) throws -> PersistentPromptCacheBlockKey {
        return try PersistentPromptCacheBlockKey.forRootBlockWithCausalInput(
            modelContract: modelContract,
            blockTokens: blockTokens,
            blockCausalInput: PersistentPromptCacheBlockCausalInput.empty());
    }

    /// Hashes the first block while binding non-token causal input introduced there.
    public static func forRootBlockWithCausalInput(
        modelContract: PersistentPromptCacheModelContract,
        blockTokens: [UInt32],
        blockCausalInput: PersistentPromptCacheBlockCausalInput
    ) throws -> PersistentPromptCacheBlockKey {
        let blockTokenCount: Int = modelContract.blockTokenCount;
        try PersistentPromptCacheBlockKey.validateBlockTokens(
            blockTokens: blockTokens, blockTokenCount: blockTokenCount);
        // Persisted tensors are reusable only when their complete storage
        // geometry agrees. Carrying the contract fingerprint in the root
        // digest prevents equal prompt tokens from crossing model revisions,
        // dtypes, layer layouts, or model-derived block lengths.
        let storageContractFingerprint: Data = modelContract.storageContractFingerprint();
        let blockHash: Data = PersistentPromptCacheBlockKey.chainHashWithCausalInput(
            parentHash: nil,
            storageContractFingerprint: storageContractFingerprint,
            blockTokens: blockTokens,
            blockCausalInput: blockCausalInput);
        guard blockTokens.count <= UInt32.max else {
            throw PersistentPromptCacheBlockKeyError.blockTokenCountOverflow;
        }
        return PersistentPromptCacheBlockKey(
            blockHashData: blockHash,
            blockIndexValue: 0,
            tokenCountValue: UInt32(blockTokens.count),
            blockTokenCountValue: blockTokenCount,
            storageContractFingerprintData: storageContractFingerprint);
    }

    /// Hashes the next block in the chain, carrying the complete storage identity forward.
    public func forChildBlock(
        blockTokens: [UInt32]
    ) throws -> PersistentPromptCacheBlockKey {
        return try self.forChildBlockWithCausalInput(
            blockTokens: blockTokens,
            blockCausalInput: PersistentPromptCacheBlockCausalInput.empty());
    }

    /// Hashes the next block with the non-token causal input introduced by that block.
    public func forChildBlockWithCausalInput(
        blockTokens: [UInt32],
        blockCausalInput: PersistentPromptCacheBlockCausalInput
    ) throws -> PersistentPromptCacheBlockKey {
        try PersistentPromptCacheBlockKey.validateBlockTokens(
            blockTokens: blockTokens, blockTokenCount: self.blockTokenCountValue);
        // Include the parent digest rather than only the child tokens so a
        // shared suffix cannot restore state produced by a divergent prefix.
        let blockHash: Data = PersistentPromptCacheBlockKey.chainHashWithCausalInput(
            parentHash: self.blockHashData,
            storageContractFingerprint: self.storageContractFingerprintData,
            blockTokens: blockTokens,
            blockCausalInput: blockCausalInput);
        guard self.blockIndexValue != UInt32.max else {
            throw PersistentPromptCacheBlockKeyError.blockIndexOverflow;
        }
        guard blockTokens.count <= UInt32.max else {
            throw PersistentPromptCacheBlockKeyError.blockTokenCountOverflow;
        }
        return PersistentPromptCacheBlockKey(
            blockHashData: blockHash,
            blockIndexValue: self.blockIndexValue + 1,
            tokenCountValue: UInt32(blockTokens.count),
            blockTokenCountValue: self.blockTokenCountValue,
            storageContractFingerprintData: self.storageContractFingerprintData);
    }

    /// The 32-byte content digest identifying this block in its chain.
    public func blockHash() -> Data {
        return self.blockHashData;
    }

    /// The zero-based position of this block in the prompt chain.
    public func blockIndex() -> UInt32 {
        return self.blockIndexValue;
    }

    /// The prompt tokens this block covers.
    public func tokenCount() -> Int {
        return Int(self.tokenCountValue);
    }

    /// The contract's full-block token count this block was hashed under.
    public func blockTokenCount() -> Int {
        return self.blockTokenCountValue;
    }

    private static func validateBlockTokens(
        blockTokens: [UInt32], blockTokenCount: Int
    ) throws {
        if blockTokens.isEmpty {
            throw PersistentPromptCacheBlockKeyError.emptyBlockTokens;
        }
        if blockTokens.count > blockTokenCount {
            throw PersistentPromptCacheBlockKeyError.blockTokenCountExceedsBlock(
                actualTokenCount: blockTokens.count,
                maximumTokenCount: blockTokenCount);
        }
    }

    private static func chainHashWithCausalInput(
        parentHash: Data?,
        storageContractFingerprint: Data,
        blockTokens: [UInt32],
        blockCausalInput: PersistentPromptCacheBlockCausalInput
    ) -> Data {
        var hasher: SHA256 = SHA256();
        if let parentHash: Data = parentHash {
            hasher.update(data: parentHash);
        } else {
            hasher.update(data: PersistentPromptCacheBlockKey.PERSISTENT_PROMPT_CACHE_ROOT_SEED);
        }
        hasher.update(data: storageContractFingerprint);
        if let blockCausalInputDigest: Data = blockCausalInput.canonicalDigest() {
            hasher.update(data: PersistentPromptCacheBlockKey
                .PERSISTENT_PROMPT_CACHE_BLOCK_CAUSAL_INPUT_DOMAIN);
            hasher.update(data: blockCausalInputDigest);
        }
        for blockToken: UInt32 in blockTokens {
            var bigEndianToken: UInt32 = blockToken.bigEndian;
            withUnsafeBytes(of: &bigEndianToken) { (tokenBuffer: UnsafeRawBufferPointer) in
                hasher.update(data: tokenBuffer);
            };
        }
        return Data(hasher.finalize());
    }
}
