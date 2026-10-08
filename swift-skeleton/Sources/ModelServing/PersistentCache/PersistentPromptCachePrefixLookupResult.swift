import Foundation;

/// The longest restorable persistent prompt-cache prefix, determined without
/// any engine dependency, port of the Rust
/// `PersistentPromptCachePrefixLookupResult`.
public struct PersistentPromptCachePrefixLookupResult: Equatable, Sendable {

    private let restoredTokenCountValue: Int;
    // This is copied so callers can retain the lookup outcome after the
    // input request buffer has been released. It is the exact suffix the
    // engine must still forward through the model.
    private let remainingTokensValue: [UInt32];
    private let lastRestoredBlockKeyValue: PersistentPromptCacheBlockKey?;
    private let restoredPartialTailBlockKeyValue: PersistentPromptCacheBlockKey?;
    private let lookupDiagnosticsValue: PersistentPromptCacheLookupDiagnostics;

    init(
        restoredTokenCount: Int,
        remainingTokens: [UInt32],
        lastRestoredBlockKey: PersistentPromptCacheBlockKey?,
        restoredPartialTailBlockKey: PersistentPromptCacheBlockKey?,
        lookupDiagnostics: PersistentPromptCacheLookupDiagnostics
    ) {
        self.restoredTokenCountValue = restoredTokenCount;
        self.remainingTokensValue = remainingTokens;
        self.lastRestoredBlockKeyValue = lastRestoredBlockKey;
        self.restoredPartialTailBlockKeyValue = restoredPartialTailBlockKey;
        self.lookupDiagnosticsValue = lookupDiagnostics;
    }

    /// The number of prompt tokens that can be restored from the persistent prompt cache.
    public var restoredTokenCount: Int {
        return self.restoredTokenCountValue;
    }

    /// The prompt tokens that still need forward processing.
    public var remainingTokens: [UInt32] {
        return self.remainingTokensValue;
    }

    /// The persistent prompt-cache block identity of the last matched block,
    /// if any. The engine uses this to chain the next block it saves during
    /// prefill: `lastRestoredBlockKey.forChildBlock(nextBlockTokens)`.
    public var lastRestoredBlockKey: PersistentPromptCacheBlockKey? {
        return self.lastRestoredBlockKeyValue;
    }

    /// The partial tail block restored on top of the last complete block, if
    /// any. A tail holds fewer tokens than one full block and is never a
    /// chain parent: the engine must keep chaining new complete blocks from
    /// `lastRestoredBlockKey`, which stays on the last complete block even
    /// when a tail was restored.
    public var restoredPartialTailBlockKey: PersistentPromptCacheBlockKey? {
        return self.restoredPartialTailBlockKeyValue;
    }

    /// Evidence describing how the lookup reached its result.
    public var diagnostics: PersistentPromptCacheLookupDiagnostics {
        return self.lookupDiagnosticsValue;
    }
}
