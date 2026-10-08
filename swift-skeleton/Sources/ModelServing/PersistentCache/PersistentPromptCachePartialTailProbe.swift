import Foundation;

/// Brute-force probe for one restorable partial tail block on top of a
/// matched prefix, port of the Rust `partial_tail_probe`. Partial tails are
/// stored off the complete-block chain: they hold fewer tokens than one full
/// block and are never chain parents. Lookup reconstructs each candidate key
/// from the prompt itself and asks the store whether that exact key exists;
/// a hash hit proves both the token span and the chain position, because
/// block hashes cover the parent digest, the causal input, and every tail
/// token.
enum PersistentPromptCachePartialTailProbe {

    /// Probes for a restorable partial tail block continuing one matched prefix.
    ///
    /// `restoredCompleteBlockKeys` holds the matched complete-block chain in
    /// prompt order. An empty chain probes a root tail keyed from the root
    /// seed; otherwise the tail continues the newest matched block. The
    /// probe walks candidate token counts longest-first and returns the
    /// first key whose sequence-state and recurrent-snapshot files both
    /// exist, so the longest stored tail under one parent wins.
    ///
    /// `tailBlockSlotIndex` is the prompt block slot the tail occupies; it
    /// selects the model-owned causal input for that slot. A missing causal
    /// input disables the probe (fail-soft, matching complete-block lookup).
    static func probeRestorablePartialTailBlock(
        modelContract: PersistentPromptCacheModelContract,
        promptTokens: [UInt32],
        restoredCompleteBlockKeys: [PersistentPromptCacheBlockKey],
        newestSnapshotIsChainTip: Bool,
        allowExactBlockBoundaryRestore: Bool,
        blockCausalInputs: [PersistentPromptCacheBlockCausalInput]?,
        kvBlockExists: (Data) -> Bool,
        recurrentSnapshotExists: (Data) -> Bool
    ) -> PersistentPromptCacheBlockKey? {
        if modelContract.hasSequenceState == false || promptTokens.isEmpty {
            return nil;
        }
        let persistentPromptCacheBlockTokenCount: Int = modelContract.blockTokenCount;
        let tailStartTokens: Int = restoredCompleteBlockKeys.count
            * persistentPromptCacheBlockTokenCount;
        guard promptTokens.count >= tailStartTokens else {
            return nil;
        }
        let tailTokenCount: Int = promptTokens.count - tailStartTokens;
        if tailTokenCount == 0 {
            return nil;
        }
        // Generation startup must keep at least one prompt token for the
        // forward pass that produces the first logits; a complete-prefix
        // consumer may consume the whole tail. A tail is also strictly
        // partial, so the longest candidate stays under one full block.
        let retainedSuffixTokenCount: Int = allowExactBlockBoundaryRestore ? 0 : 1;
        let maximumTailTokenCount: Int = min(
            max(tailTokenCount - retainedSuffixTokenCount, 0),
            persistentPromptCacheBlockTokenCount - 1);
        if maximumTailTokenCount == 0 {
            return nil;
        }
        if restoredCompleteBlockKeys.isEmpty == false {
            // A tail is restorable only when the newest recurrent snapshot
            // sits on the chain tip: otherwise the restored state cannot
            // reach the tail's parent. And when the uncached suffix still
            // contains complete blocks, the final partial segment chains
            // from a block this lookup did not reach, so no tail under the
            // matched tip can exist.
            if tailTokenCount >= persistentPromptCacheBlockTokenCount
                || newestSnapshotIsChainTip == false {
                return nil;
            }
        }
        let tailBlockSlotIndex: Int = restoredCompleteBlockKeys.count;
        let emptyBlockCausalInput: PersistentPromptCacheBlockCausalInput =
            PersistentPromptCacheBlockCausalInput.empty();
        let tailBlockCausalInput: PersistentPromptCacheBlockCausalInput?;
        switch blockCausalInputs {
        case .none:
            tailBlockCausalInput = emptyBlockCausalInput;
        case .some(let causalInputs):
            guard causalInputs.indices.contains(tailBlockSlotIndex) else {
                return nil;
            }
            tailBlockCausalInput = causalInputs[tailBlockSlotIndex];
        }
        guard let tailBlockCausalInput: PersistentPromptCacheBlockCausalInput = tailBlockCausalInput
        else {
            return nil;
        }
        let parentBlockKey: PersistentPromptCacheBlockKey? = restoredCompleteBlockKeys.last;
        for tailCandidateTokenCount in stride(
            from: maximumTailTokenCount, through: 1, by: -1) {
            let tailTokens: [UInt32] = Array(promptTokens[
                tailStartTokens..<(tailStartTokens + tailCandidateTokenCount)]);
            let tailBlockKey: PersistentPromptCacheBlockKey?;
            do {
                if let parentBlockKey: PersistentPromptCacheBlockKey = parentBlockKey {
                    tailBlockKey = try parentBlockKey.forChildBlockWithCausalInput(
                        blockTokens: tailTokens, blockCausalInput: tailBlockCausalInput);
                } else {
                    tailBlockKey = try PersistentPromptCacheBlockKey.forRootBlockWithCausalInput(
                        modelContract: modelContract,
                        blockTokens: tailTokens,
                        blockCausalInput: tailBlockCausalInput);
                }
            } catch {
                tailBlockKey = nil;
            }
            guard let candidateBlockKey: PersistentPromptCacheBlockKey = tailBlockKey else {
                continue;
            }
            let tailBlockHash: Data = candidateBlockKey.blockHash();
            if kvBlockExists(tailBlockHash) && recurrentSnapshotExists(tailBlockHash) {
                return candidateBlockKey;
            }
        }
        return nil;
    }
}
