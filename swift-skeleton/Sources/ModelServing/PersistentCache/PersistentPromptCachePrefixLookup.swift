import Foundation;

/// Pure lookup for the longest restorable prompt prefix, port of the Rust
/// `PersistentPromptCachePrefixLookup`. This is the decision layer the
/// engine calls before any allocation: it hashes the prompt's persistent
/// prompt-cache blocks in chain order, finds the longest contiguous KV
/// prefix, then walks backward to find the newest recurrent snapshot that
/// can safely seed continuation. It enforces the safety invariant that at
/// least one prompt token always remains for forward processing, even when
/// the prompt ends exactly on a block boundary.
public enum PersistentPromptCachePrefixLookup {

    /// Computes the longest restorable prefix of one prompt.
    ///
    /// The caller supplies separate KV-block and recurrent-snapshot
    /// predicates to query the persistent prompt cache without coupling this
    /// pure decision layer to the filesystem.
    public static func forPrompt(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCachePrefixLookupResult {
        return PersistentPromptCachePrefixLookup.forPromptWithBlockCausalInputsAndBoundaryPolicy(
            modelContract: modelContract,
            promptTokens: promptTokens,
            blockCausalInputs: nil,
            allowExactBlockBoundaryRestore: false,
            probePartialTailBlocks: false,
            kvBlockExists: kvBlockExists,
            recurrentSnapshotExists: recurrentSnapshotExists);
    }

    /// Computes the longest restorable complete prefix, including an exact
    /// block boundary. This variant is for a private decoder state owner
    /// that will continue with another model-side operation; it does not
    /// need to retain a final prompt token for logits production.
    public static func forCompletePrefix(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCachePrefixLookupResult {
        return PersistentPromptCachePrefixLookup.forPromptWithBlockCausalInputsAndBoundaryPolicy(
            modelContract: modelContract,
            promptTokens: promptTokens,
            blockCausalInputs: nil,
            allowExactBlockBoundaryRestore: true,
            probePartialTailBlocks: false,
            kvBlockExists: kvBlockExists,
            recurrentSnapshotExists: recurrentSnapshotExists);
    }

    /// Computes the longest restorable prefix with model-owned causal input
    /// for each block.
    public static func forPromptWithBlockCausalInputs(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        blockCausalInputs: [PersistentPromptCacheBlockCausalInput],
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCachePrefixLookupResult {
        return PersistentPromptCachePrefixLookup.forPromptWithBlockCausalInputsAndBoundaryPolicy(
            modelContract: modelContract,
            promptTokens: promptTokens,
            blockCausalInputs: blockCausalInputs.isEmpty ? nil : blockCausalInputs,
            allowExactBlockBoundaryRestore: false,
            probePartialTailBlocks: false,
            kvBlockExists: kvBlockExists,
            recurrentSnapshotExists: recurrentSnapshotExists);
    }

    /// Computes the longest restorable prefix, additionally reusing one
    /// stored partial tail block. Multi-turn prompts grow by a partial final
    /// block between turns: when the previous turn published that partial
    /// suffix as a tail block, this lookup restores it on top of the matched
    /// complete-block chain so the next turn prefills only the uncached
    /// suffix.
    public static func forPromptWithPartialTailBlockRecovery(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        blockCausalInputs: [PersistentPromptCacheBlockCausalInput],
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCachePrefixLookupResult {
        // An empty causal-input array means "no model-owned causal input"
        // for the existing constructors; preserve that meaning here too.
        return PersistentPromptCachePrefixLookup.forPromptWithBlockCausalInputsAndBoundaryPolicy(
            modelContract: modelContract,
            promptTokens: promptTokens,
            blockCausalInputs: blockCausalInputs.isEmpty ? nil : blockCausalInputs,
            allowExactBlockBoundaryRestore: false,
            probePartialTailBlocks: true,
            kvBlockExists: kvBlockExists,
            recurrentSnapshotExists: recurrentSnapshotExists);
    }

    private static func forPromptWithBlockCausalInputsAndBoundaryPolicy(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        blockCausalInputs: [PersistentPromptCacheBlockCausalInput]?,
        allowExactBlockBoundaryRestore: Bool,
        probePartialTailBlocks: Bool,
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCachePrefixLookupResult {
        let persistentPromptCacheBlockTokenCount: Int = modelContract.blockTokenCount;
        // The contract resolver guarantees a nonzero block length. Keeping
        // the calculation here contract-driven makes lookup agree with
        // capture on models whose state geometry selects a different
        // boundary size than another artifact on the same machine.
        let completePromptBlockCount: Int =
            promptTokens.count / persistentPromptCacheBlockTokenCount;
        // An exact block-boundary prompt must retain its final block for a
        // forward pass. The final prompt token produces the logits used to
        // begin decode, while restoring only recurrent state cannot
        // substitute for that model computation.
        let maximumRestorableBlockCount: Int;
        if allowExactBlockBoundaryRestore {
            maximumRestorableBlockCount = completePromptBlockCount;
        } else if promptTokens.count % persistentPromptCacheBlockTokenCount == 0 {
            maximumRestorableBlockCount = max(completePromptBlockCount - 1, 0);
        } else {
            maximumRestorableBlockCount = completePromptBlockCount;
        }
        var lookupDiagnostics: PersistentPromptCacheLookupDiagnostics =
            PersistentPromptCacheLookupDiagnostics(
                completePromptBlockCount: completePromptBlockCount,
                maximumRestorableBlockCount: maximumRestorableBlockCount);
        if maximumRestorableBlockCount == 0 {
            // A prompt shorter than one complete block can still reuse a
            // root tail published by a previous turn with the same short
            // prefix.
            if probePartialTailBlocks {
                if let restoredPartialTailBlockKey: PersistentPromptCacheBlockKey =
                    PersistentPromptCachePartialTailProbe.probeRestorablePartialTailBlock(
                        modelContract: modelContract,
                        promptTokens: promptTokens,
                        restoredCompleteBlockKeys: [],
                        newestSnapshotIsChainTip: false,
                        allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
                        blockCausalInputs: blockCausalInputs,
                        kvBlockExists: kvBlockExists,
                        recurrentSnapshotExists: recurrentSnapshotExists) {
                    let restoredTokenCount: Int = restoredPartialTailBlockKey.tokenCount();
                    lookupDiagnostics = lookupDiagnostics
                        .recordingRestoredPartialTailBlockTokenCount(restoredTokenCount);
                    return PersistentPromptCachePrefixLookupResult(
                        restoredTokenCount: restoredTokenCount,
                        remainingTokens: Array(promptTokens[restoredTokenCount...]),
                        lastRestoredBlockKey: nil,
                        restoredPartialTailBlockKey: restoredPartialTailBlockKey,
                        lookupDiagnostics: lookupDiagnostics);
                }
            }
            lookupDiagnostics = lookupDiagnostics.recordingMissReason(
                .promptTooShortForPersistentPromptCache);
            return PersistentPromptCachePrefixLookup.cacheMissLookupResult(
                promptTokens: promptTokens, lookupDiagnostics: lookupDiagnostics);
        }
        // The key chain is constructed in prompt order. A hit for block N is
        // meaningful only if every predecessor also matched, which prevents
        // reuse across prompts that diverge in an earlier block.
        var matchedBlockKeys: [PersistentPromptCacheBlockKey] = [];
        matchedBlockKeys.reserveCapacity(maximumRestorableBlockCount);
        var parentBlockKey: PersistentPromptCacheBlockKey? = nil;
        let emptyBlockCausalInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput.empty();
        for blockIndex: Int in 0..<maximumRestorableBlockCount {
            let blockStart: Int = blockIndex * persistentPromptCacheBlockTokenCount;
            let blockEnd: Int = blockStart + persistentPromptCacheBlockTokenCount;
            let blockTokens: [UInt32] = Array(promptTokens[blockStart..<blockEnd]);
            let blockCausalInput: PersistentPromptCacheBlockCausalInput;
            switch blockCausalInputs {
            case .none:
                blockCausalInput = emptyBlockCausalInput;
            case .some(let causalInputs):
                guard let suppliedCausalInput: PersistentPromptCacheBlockCausalInput =
                    causalInputs.indices.contains(blockIndex) ? causalInputs[blockIndex] : nil
                else {
                    lookupDiagnostics = lookupDiagnostics
                        .recordingFirstMissingSequenceStateBlock(index: blockIndex, hash: nil);
                    return PersistentPromptCachePrefixLookup.finishLookup(
                        modelContract: modelContract,
                        promptTokens: promptTokens,
                        matchedBlockKeys: matchedBlockKeys,
                        blockCausalInputs: blockCausalInputs,
                        allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
                        probePartialTailBlocks: probePartialTailBlocks,
                        kvBlockExists: kvBlockExists,
                        recurrentSnapshotExists: recurrentSnapshotExists,
                        lookupDiagnostics: lookupDiagnostics);
                }
                blockCausalInput = suppliedCausalInput;
            }
            let persistentPromptCacheBlockKey: PersistentPromptCacheBlockKey?;
            do {
                if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
                    persistentPromptCacheBlockKey = try parentBlockKey
                        .forChildBlockWithCausalInput(
                            blockTokens: blockTokens, blockCausalInput: blockCausalInput);
                } else {
                    persistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
                        .forRootBlockWithCausalInput(
                            modelContract: modelContract,
                            blockTokens: blockTokens,
                            blockCausalInput: blockCausalInput);
                }
            } catch {
                // Invalid key construction is equivalent to a missing block
                // here. This pure decision layer intentionally remains
                // fail-soft; the engine can always process the complete
                // prompt cold.
                persistentPromptCacheBlockKey = nil;
            }
            guard let constructedBlockKey: PersistentPromptCacheBlockKey = persistentPromptCacheBlockKey
            else {
                lookupDiagnostics = lookupDiagnostics
                    .recordingFirstMissingSequenceStateBlock(index: blockIndex, hash: nil);
                return PersistentPromptCachePrefixLookup.finishLookup(
                    modelContract: modelContract,
                    promptTokens: promptTokens,
                    matchedBlockKeys: matchedBlockKeys,
                    blockCausalInputs: blockCausalInputs,
                    allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
                    probePartialTailBlocks: probePartialTailBlocks,
                    kvBlockExists: kvBlockExists,
                    recurrentSnapshotExists: recurrentSnapshotExists,
                    lookupDiagnostics: lookupDiagnostics);
            }
            if modelContract.hasSequenceState
                && kvBlockExists(constructedBlockKey.blockHash()) == false {
                lookupDiagnostics = lookupDiagnostics.recordingFirstMissingSequenceStateBlock(
                    index: blockIndex, hash: constructedBlockKey.blockHash());
                return PersistentPromptCachePrefixLookup.finishLookup(
                    modelContract: modelContract,
                    promptTokens: promptTokens,
                    matchedBlockKeys: matchedBlockKeys,
                    blockCausalInputs: blockCausalInputs,
                    allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
                    probePartialTailBlocks: probePartialTailBlocks,
                    kvBlockExists: kvBlockExists,
                    recurrentSnapshotExists: recurrentSnapshotExists,
                    lookupDiagnostics: lookupDiagnostics);
            }
            // Boundary-only models still construct the deterministic chain,
            // but do not require a sequence-state file. Their latest complete
            // boundary snapshot is sufficient for restore, whereas hybrid
            // models require both forms of state at the same boundary.
            parentBlockKey = constructedBlockKey;
            matchedBlockKeys.append(constructedBlockKey);
        }
        return PersistentPromptCachePrefixLookup.finishLookup(
            modelContract: modelContract,
            promptTokens: promptTokens,
            matchedBlockKeys: matchedBlockKeys,
            blockCausalInputs: blockCausalInputs,
            allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
            probePartialTailBlocks: probePartialTailBlocks,
            kvBlockExists: kvBlockExists,
            recurrentSnapshotExists: recurrentSnapshotExists,
            lookupDiagnostics: lookupDiagnostics);
    }

    /// Resolves the newest usable recurrent snapshot over the matched chain
    /// and assembles the final lookup outcome. Splitting this tail out keeps
    /// the chain-walk loop linear and mirrors the Rust control flow exactly.
    private static func finishLookup(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        matchedBlockKeys: [PersistentPromptCacheBlockKey],
        blockCausalInputs: [PersistentPromptCacheBlockCausalInput]?,
        allowExactBlockBoundaryRestore: Bool,
        probePartialTailBlocks: Bool,
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool,
        lookupDiagnostics: PersistentPromptCacheLookupDiagnostics
    ) -> PersistentPromptCachePrefixLookupResult {
        var lookupDiagnostics: PersistentPromptCacheLookupDiagnostics = lookupDiagnostics;
        if modelContract.hasSequenceState {
            lookupDiagnostics = lookupDiagnostics.recordingMatchedSequenceStateBlockCount(
                matchedBlockKeys.count);
        }
        // Walk backward for the newest snapshot that can safely seed
        // continuation; boundary-only models take the newest complete block.
        var recurrentSnapshotBlockIndex: Int? = nil;
        var restoredBlockKey: PersistentPromptCacheBlockKey? = nil;
        for (blockIndex, blockKey) in matchedBlockKeys.enumerated().reversed() {
            if modelContract.hasBoundaryState == false
                || recurrentSnapshotExists(blockKey.blockHash()) {
                recurrentSnapshotBlockIndex = blockIndex;
                restoredBlockKey = blockKey;
                break;
            }
        }
        guard let recurrentSnapshotBlockIndex: Int = recurrentSnapshotBlockIndex,
            let restoredBlockKey: PersistentPromptCacheBlockKey = restoredBlockKey
        else {
            let missReason: PersistentPromptCacheMissReason;
            switch (modelContract.hasSequenceState,
                lookupDiagnostics.firstMissingSequenceStateBlockIndex) {
            case (true, .some(0)):
                missReason = .rootSequenceStateBlockMissing;
            default:
                missReason = .boundaryStateSnapshotMissing;
            }
            lookupDiagnostics = lookupDiagnostics.recordingMissReason(missReason);
            return PersistentPromptCachePrefixLookup.cacheMissLookupResult(
                promptTokens: promptTokens, lookupDiagnostics: lookupDiagnostics);
        }
        lookupDiagnostics = lookupDiagnostics
            .recordingNewestBoundaryStateSnapshotBlockIndex(recurrentSnapshotBlockIndex);
        let restoredBlockCount: Int = Int(restoredBlockKey.blockIndex()) + 1;
        let restoredCompleteBlockTokenCount: Int =
            restoredBlockCount * modelContract.blockTokenCount;
        if restoredCompleteBlockTokenCount == 0
            || restoredCompleteBlockTokenCount > promptTokens.count {
            lookupDiagnostics = lookupDiagnostics.recordingMissReason(
                .boundaryStateSnapshotMissing);
            return PersistentPromptCachePrefixLookup.cacheMissLookupResult(
                promptTokens: promptTokens, lookupDiagnostics: lookupDiagnostics);
        }
        // Multi-turn prompts grow by a partial final block. When the
        // previous turn published that suffix as a tail, restore it on top
        // of the complete-block chain so only the uncached suffix is
        // prefilled.
        let restoredPartialTailBlockKey: PersistentPromptCacheBlockKey? =
            probePartialTailBlocks
            ? PersistentPromptCachePartialTailProbe.probeRestorablePartialTailBlock(
                modelContract: modelContract,
                promptTokens: promptTokens,
                restoredCompleteBlockKeys: matchedBlockKeys,
                newestSnapshotIsChainTip: recurrentSnapshotBlockIndex + 1
                    == matchedBlockKeys.count,
                allowExactBlockBoundaryRestore: allowExactBlockBoundaryRestore,
                blockCausalInputs: blockCausalInputs,
                kvBlockExists: kvBlockExists,
                recurrentSnapshotExists: recurrentSnapshotExists)
            : nil;
        if let restoredPartialTailBlockKey: PersistentPromptCacheBlockKey = restoredPartialTailBlockKey {
            lookupDiagnostics = lookupDiagnostics
                .recordingRestoredPartialTailBlockTokenCount(
                    restoredPartialTailBlockKey.tokenCount());
        }
        let restoredTokenCount: Int = restoredCompleteBlockTokenCount
            + (restoredPartialTailBlockKey?.tokenCount() ?? 0);
        // Return the untouched suffix rather than a block-rounded slice. The
        // final partial block and the required final token both belong to
        // the normal prefill path.
        return PersistentPromptCachePrefixLookupResult(
            restoredTokenCount: restoredTokenCount,
            remainingTokens: Array(promptTokens[restoredTokenCount...]),
            lastRestoredBlockKey: restoredBlockKey,
            restoredPartialTailBlockKey: restoredPartialTailBlockKey,
            lookupDiagnostics: lookupDiagnostics);
    }

    private static func cacheMissLookupResult(
        promptTokens: [UInt32],
        lookupDiagnostics: PersistentPromptCacheLookupDiagnostics
    ) -> PersistentPromptCachePrefixLookupResult {
        return PersistentPromptCachePrefixLookupResult(
            restoredTokenCount: 0,
            remainingTokens: promptTokens,
            lastRestoredBlockKey: nil,
            restoredPartialTailBlockKey: nil,
            lookupDiagnostics: lookupDiagnostics);
    }
}
