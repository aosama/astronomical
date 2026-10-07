import Foundation;

/// Bounded per-request evidence explaining persistent prompt-cache behavior.
///
/// These types cross the worker/supervisor boundary and can enter local JSONL
/// performance logs. They intentionally contain counters, enums, and a short hash
/// prefix only—never prompts, complete hashes, model paths, or tensor contents.

public enum WorkerPersistentPromptCacheLookupOutcome: Equatable {
    case hit;
    case miss;

    private static let expectedVariantNames: Array<String> = ["hit", "miss"];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .hit: return .string("hit");
        case .miss: return .string("miss");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheLookupOutcome {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "hit": return .hit;
        case "miss": return .miss;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(WorkerPersistentPromptCacheLookupOutcome.expectedVariantNames))");
        }
    }
}

public enum WorkerPersistentPromptCacheMissReason: Equatable {
    /// The prompt does not contain one complete cache block.
    case promptTooShortForPersistentPromptCache;
    /// The content-addressed root sequence block was not durable.
    case rootSequenceStateBlockMissing;
    /// Sequence ancestry matched, but no required restart boundary was available.
    case boundaryStateSnapshotMissing;

    private static let expectedVariantNames: Array<String> = [
        "prompt_too_short_for_persistent_prompt_cache",
        "root_sequence_state_block_missing",
        "boundary_state_snapshot_missing",
    ];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case .promptTooShortForPersistentPromptCache: return .string("prompt_too_short_for_persistent_prompt_cache");
        case .rootSequenceStateBlockMissing: return .string("root_sequence_state_block_missing");
        case .boundaryStateSnapshotMissing: return .string("boundary_state_snapshot_missing");
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheMissReason {
        let wireName = try JsonWireValue.extractString(wireValue);
        switch wireName {
        case "prompt_too_short_for_persistent_prompt_cache": return .promptTooShortForPersistentPromptCache;
        case "root_sequence_state_block_missing": return .rootSequenceStateBlockMissing;
        case "boundary_state_snapshot_missing": return .boundaryStateSnapshotMissing;
        default:
            throw JsonWireProblem.malformedDocument(problem: "unknown variant `\(wireName)`, expected one of \(JsonWireProblem.formattedFieldList(WorkerPersistentPromptCacheMissReason.expectedVariantNames))");
        }
    }
}

/// Count and byte total for one startup-cleanup reason.
public struct WorkerPersistentPromptCacheStartupCleanupCategory: Equatable {
    /// Standalone files removed for this reason.
    public let artifactCount: UInt64;
    /// Atomic block directories removed for this reason.
    public let blockCount: UInt64;
    /// Total non-directory bytes removed for this reason.
    public let byteCount: UInt64;

    public init(artifactCount: UInt64, blockCount: UInt64, byteCount: UInt64) {
        self.artifactCount = artifactCount;
        self.blockCount = blockCount;
        self.byteCount = byteCount;
    }

    internal static let wireFieldNames: Array<String> = ["artifact_count", "block_count", "byte_count"];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "artifact_count", value: .unsignedInteger(self.artifactCount));
        wireObject.appendEntry(key: "block_count", value: .unsignedInteger(self.blockCount));
        wireObject.appendEntry(key: "byte_count", value: .unsignedInteger(self.byteCount));
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheStartupCleanupCategory {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedCategory = WorkerPersistentPromptCacheStartupCleanupCategory(
            artifactCount: try wireObject.decodeUInt64(fieldName: "artifact_count"),
            blockCount: try wireObject.decodeUInt64(fieldName: "block_count"),
            byteCount: try wireObject.decodeUInt64(fieldName: "byte_count"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerPersistentPromptCacheStartupCleanupCategory.wireFieldNames);
        return parsedCategory;
    }
}

/// Bounded reason-separated evidence retained from prompt-cache startup.
public struct WorkerPersistentPromptCacheStartupCleanupEvidence: Equatable {
    public let interruptedTransactionRecovery: WorkerPersistentPromptCacheStartupCleanupCategory;
    public let obsoleteFormat: WorkerPersistentPromptCacheStartupCleanupCategory;
    public let corruptCurrentFormat: WorkerPersistentPromptCacheStartupCleanupCategory;
    public let quotaEviction: WorkerPersistentPromptCacheStartupCleanupCategory;

    public init(
        interruptedTransactionRecovery: WorkerPersistentPromptCacheStartupCleanupCategory,
        obsoleteFormat: WorkerPersistentPromptCacheStartupCleanupCategory,
        corruptCurrentFormat: WorkerPersistentPromptCacheStartupCleanupCategory,
        quotaEviction: WorkerPersistentPromptCacheStartupCleanupCategory
    ) {
        self.interruptedTransactionRecovery = interruptedTransactionRecovery;
        self.obsoleteFormat = obsoleteFormat;
        self.corruptCurrentFormat = corruptCurrentFormat;
        self.quotaEviction = quotaEviction;
    }

    internal static let wireFieldNames: Array<String> = [
        "interrupted_transaction_recovery", "obsolete_format", "corrupt_current_format", "quota_eviction",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "interrupted_transaction_recovery", value: self.interruptedTransactionRecovery.wireValue());
        wireObject.appendEntry(key: "obsolete_format", value: self.obsoleteFormat.wireValue());
        wireObject.appendEntry(key: "corrupt_current_format", value: self.corruptCurrentFormat.wireValue());
        wireObject.appendEntry(key: "quota_eviction", value: self.quotaEviction.wireValue());
        return .object(wireObject);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheStartupCleanupEvidence {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedEvidence = WorkerPersistentPromptCacheStartupCleanupEvidence(
            interruptedTransactionRecovery: try WorkerPersistentPromptCacheStartupCleanupCategory.fromWireValue(try wireObject.requireObjectValue(fieldName: "interrupted_transaction_recovery")),
            obsoleteFormat: try WorkerPersistentPromptCacheStartupCleanupCategory.fromWireValue(try wireObject.requireObjectValue(fieldName: "obsolete_format")),
            corruptCurrentFormat: try WorkerPersistentPromptCacheStartupCleanupCategory.fromWireValue(try wireObject.requireObjectValue(fieldName: "corrupt_current_format")),
            quotaEviction: try WorkerPersistentPromptCacheStartupCleanupCategory.fromWireValue(try wireObject.requireObjectValue(fieldName: "quota_eviction")));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerPersistentPromptCacheStartupCleanupEvidence.wireFieldNames);
        return parsedEvidence;
    }
}

/// Short hexadecimal prefix of one expected cache-block hash. Serde treats the
/// wrapper transparently: the wire form is the bare prefix string.
///
/// Eight bytes (16 hexadecimal characters) are enough to correlate local
/// diagnostics while avoiding publication of complete content identity.
public struct WorkerPersistentPromptCacheExpectedBlockHashPrefix: Equatable {
    public let prefixText: String;

    public init(prefixText: String) {
        self.prefixText = prefixText;
    }

    /// Builds the prefix from a 32-byte block hash; requires at least eight bytes.
    public static func fromBlockHash(blockHash: Array<UInt8>) -> WorkerPersistentPromptCacheExpectedBlockHashPrefix {
        let prefixBytes = blockHash.prefix(8);
        let prefixCharacters = prefixBytes.map({ (blockHashByte: UInt8) -> String in
            String(format: "%02x", blockHashByte);
        });
        return WorkerPersistentPromptCacheExpectedBlockHashPrefix(prefixText: prefixCharacters.joined());
    }

    public func asStr() -> String {
        return self.prefixText;
    }

    internal func wireValue() -> JsonWireValue {
        return .string(self.prefixText);
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheExpectedBlockHashPrefix {
        let hashPrefix = try JsonWireValue.extractString(wireValue);
        let hashPrefixBytes = Array(hashPrefix.utf8);
        let isWithinHexadecimalAlphabet = hashPrefixBytes.count == 16
            && hashPrefixBytes.allSatisfy({ (hashCharacterByte: UInt8) -> Bool in
                (hashCharacterByte >= UInt8(ascii: "0") && hashCharacterByte <= UInt8(ascii: "9"))
                    || (hashCharacterByte >= UInt8(ascii: "a") && hashCharacterByte <= UInt8(ascii: "f"));
            });
        guard isWithinHexadecimalAlphabet else {
            throw JsonWireProblem.malformedDocument(problem: "expected block hash prefix must be 16 lowercase hexadecimal characters");
        }
        return WorkerPersistentPromptCacheExpectedBlockHashPrefix(prefixText: hashPrefix);
    }
}

public struct WorkerPersistentPromptCacheRequestDiagnostics: Equatable {
    /// Final user-visible classification after longest-prefix lookup.
    public let lookupOutcome: WorkerPersistentPromptCacheLookupOutcome;
    /// Immutable model-derived block geometry used for this request.
    public let blockTokenCount: UInt64;
    /// Number of complete blocks represented by the prompt.
    public let completePromptBlockCount: UInt64;
    /// Complete blocks eligible after retaining the final generation-start token.
    public let maximumRestorableBlockCount: UInt64;
    /// Consecutive sequence-state blocks found from root.
    public let matchedSequenceStateBlockCount: UInt64;
    /// Blocks actually reconstructed after boundary selection.
    public let restoredBlockCount: UInt64;
    /// Token count of the restored partial tail block, when one matched.
    public let partialTailBlockTokenCount: UInt64?;
    /// First expected sequence block absent from the durable chain, if any.
    public let firstMissingSequenceStateBlockIndex: UInt64?;
    public let missReason: WorkerPersistentPromptCacheMissReason?;
    /// Bounded correlation hint for the first expected missing block.
    public let expectedBlockHashPrefix: WorkerPersistentPromptCacheExpectedBlockHashPrefix?;
    /// Startup cleanup that can explain this first structural cold miss.
    public let startupCleanupEvidence: WorkerPersistentPromptCacheStartupCleanupEvidence?;
    /// Blocks physically committed by this request; idempotent reuse is excluded.
    public var publishedBlockCount: UInt64;
    /// Allocator cache released before direct tensor materialization.
    public let allocatorBytesClearedForPublication: UInt64;
    /// Pageable expert payload evicted to satisfy publication memory pressure.
    public let expertBytesReclaimedForPublication: UInt64;
    /// Expert payload reclaimed to admit cache reconstruction and remaining context.
    public let expertBytesReclaimedForRestore: UInt64;

    public init(
        lookupOutcome: WorkerPersistentPromptCacheLookupOutcome,
        blockTokenCount: UInt64,
        completePromptBlockCount: UInt64,
        maximumRestorableBlockCount: UInt64,
        matchedSequenceStateBlockCount: UInt64,
        restoredBlockCount: UInt64,
        partialTailBlockTokenCount: UInt64?,
        firstMissingSequenceStateBlockIndex: UInt64?,
        missReason: WorkerPersistentPromptCacheMissReason?,
        expectedBlockHashPrefix: WorkerPersistentPromptCacheExpectedBlockHashPrefix?,
        startupCleanupEvidence: WorkerPersistentPromptCacheStartupCleanupEvidence?,
        publishedBlockCount: UInt64,
        allocatorBytesClearedForPublication: UInt64,
        expertBytesReclaimedForPublication: UInt64,
        expertBytesReclaimedForRestore: UInt64
    ) {
        self.lookupOutcome = lookupOutcome;
        self.blockTokenCount = blockTokenCount;
        self.completePromptBlockCount = completePromptBlockCount;
        self.maximumRestorableBlockCount = maximumRestorableBlockCount;
        self.matchedSequenceStateBlockCount = matchedSequenceStateBlockCount;
        self.restoredBlockCount = restoredBlockCount;
        self.partialTailBlockTokenCount = partialTailBlockTokenCount;
        self.firstMissingSequenceStateBlockIndex = firstMissingSequenceStateBlockIndex;
        self.missReason = missReason;
        self.expectedBlockHashPrefix = expectedBlockHashPrefix;
        self.startupCleanupEvidence = startupCleanupEvidence;
        self.publishedBlockCount = publishedBlockCount;
        self.allocatorBytesClearedForPublication = allocatorBytesClearedForPublication;
        self.expertBytesReclaimedForPublication = expertBytesReclaimedForPublication;
        self.expertBytesReclaimedForRestore = expertBytesReclaimedForRestore;
    }

    /// Counts one physically committed block without overflow (Rust saturating_add).
    public mutating func recordPublishedBlock() {
        if self.publishedBlockCount != UInt64.max {
            self.publishedBlockCount += 1;
        }
    }

    internal static let wireFieldNames: Array<String> = [
        "lookup_outcome", "block_token_count", "complete_prompt_block_count",
        "maximum_restorable_block_count", "matched_sequence_state_block_count",
        "restored_block_count", "partial_tail_block_token_count",
        "first_missing_sequence_state_block_index", "miss_reason",
        "expected_block_hash_prefix", "startup_cleanup_evidence", "published_block_count",
        "allocator_bytes_cleared_for_publication", "expert_bytes_reclaimed_for_publication",
        "expert_bytes_reclaimed_for_restore",
    ];

    internal func wireValue() -> JsonWireValue {
        var wireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "lookup_outcome", value: self.lookupOutcome.wireValue());
        wireObject.appendEntry(key: "block_token_count", value: .unsignedInteger(self.blockTokenCount));
        wireObject.appendEntry(key: "complete_prompt_block_count", value: .unsignedInteger(self.completePromptBlockCount));
        wireObject.appendEntry(key: "maximum_restorable_block_count", value: .unsignedInteger(self.maximumRestorableBlockCount));
        wireObject.appendEntry(key: "matched_sequence_state_block_count", value: .unsignedInteger(self.matchedSequenceStateBlockCount));
        wireObject.appendEntry(key: "restored_block_count", value: .unsignedInteger(self.restoredBlockCount));
        wireObject.appendEntry(key: "partial_tail_block_token_count", value: WorkerPersistentPromptCacheRequestDiagnostics.optionalUInt64WireValue(self.partialTailBlockTokenCount));
        wireObject.appendEntry(key: "first_missing_sequence_state_block_index", value: WorkerPersistentPromptCacheRequestDiagnostics.optionalUInt64WireValue(self.firstMissingSequenceStateBlockIndex));
        wireObject.appendEntry(key: "miss_reason", value: WorkerPersistentPromptCacheRequestDiagnostics.optionalMissReasonWireValue(self.missReason));
        wireObject.appendEntry(key: "expected_block_hash_prefix", value: WorkerPersistentPromptCacheRequestDiagnostics.optionalBlockHashPrefixWireValue(self.expectedBlockHashPrefix));
        wireObject.appendEntry(key: "startup_cleanup_evidence", value: WorkerPersistentPromptCacheRequestDiagnostics.optionalStartupCleanupEvidenceWireValue(self.startupCleanupEvidence));
        wireObject.appendEntry(key: "published_block_count", value: .unsignedInteger(self.publishedBlockCount));
        wireObject.appendEntry(key: "allocator_bytes_cleared_for_publication", value: .unsignedInteger(self.allocatorBytesClearedForPublication));
        wireObject.appendEntry(key: "expert_bytes_reclaimed_for_publication", value: .unsignedInteger(self.expertBytesReclaimedForPublication));
        wireObject.appendEntry(key: "expert_bytes_reclaimed_for_restore", value: .unsignedInteger(self.expertBytesReclaimedForRestore));
        return .object(wireObject);
    }

    /// The serde-shaped object the supervisor embeds into the local
    /// `performance.jsonl` row; identical to the IPC wire shape because the
    /// Rust record serializes the same struct into both.
    public func performanceLogWireValue() -> JsonWireValue {
        return self.wireValue();
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerPersistentPromptCacheRequestDiagnostics {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        let parsedDiagnostics = WorkerPersistentPromptCacheRequestDiagnostics(
            lookupOutcome: try WorkerPersistentPromptCacheLookupOutcome.fromWireValue(try wireObject.requireObjectValue(fieldName: "lookup_outcome")),
            blockTokenCount: try wireObject.decodeUInt64(fieldName: "block_token_count"),
            completePromptBlockCount: try wireObject.decodeUInt64(fieldName: "complete_prompt_block_count"),
            maximumRestorableBlockCount: try wireObject.decodeUInt64(fieldName: "maximum_restorable_block_count"),
            matchedSequenceStateBlockCount: try wireObject.decodeUInt64(fieldName: "matched_sequence_state_block_count"),
            restoredBlockCount: try wireObject.decodeUInt64(fieldName: "restored_block_count"),
            partialTailBlockTokenCount: try wireObject.decodeOptionalUInt64(fieldName: "partial_tail_block_token_count"),
            firstMissingSequenceStateBlockIndex: try wireObject.decodeOptionalUInt64(fieldName: "first_missing_sequence_state_block_index"),
            missReason: try WorkerPersistentPromptCacheRequestDiagnostics.decodeOptionalMissReason(wireObject: wireObject),
            expectedBlockHashPrefix: try WorkerPersistentPromptCacheRequestDiagnostics.decodeOptionalBlockHashPrefix(wireObject: wireObject),
            startupCleanupEvidence: try WorkerPersistentPromptCacheRequestDiagnostics.decodeOptionalStartupCleanupEvidence(wireObject: wireObject),
            publishedBlockCount: try wireObject.decodeUInt64(fieldName: "published_block_count"),
            allocatorBytesClearedForPublication: try wireObject.decodeUInt64(fieldName: "allocator_bytes_cleared_for_publication"),
            expertBytesReclaimedForPublication: try wireObject.decodeUInt64(fieldName: "expert_bytes_reclaimed_for_publication"),
            expertBytesReclaimedForRestore: try wireObject.decodeUInt64(fieldName: "expert_bytes_reclaimed_for_restore"));
        try wireObject.rejectUnknownFields(allowedFieldNames: WorkerPersistentPromptCacheRequestDiagnostics.wireFieldNames);
        return parsedDiagnostics;
    }

    private static func decodeOptionalMissReason(wireObject: JsonWireObject) throws -> WorkerPersistentPromptCacheMissReason? {
        let missReasonWireValue = try wireObject.requireObjectValue(fieldName: "miss_reason");
        if missReasonWireValue.isNull {
            return nil;
        }
        return try WorkerPersistentPromptCacheMissReason.fromWireValue(missReasonWireValue);
    }

    private static func decodeOptionalBlockHashPrefix(wireObject: JsonWireObject) throws -> WorkerPersistentPromptCacheExpectedBlockHashPrefix? {
        let prefixWireValue = try wireObject.requireObjectValue(fieldName: "expected_block_hash_prefix");
        if prefixWireValue.isNull {
            return nil;
        }
        return try WorkerPersistentPromptCacheExpectedBlockHashPrefix.fromWireValue(prefixWireValue);
    }

    private static func decodeOptionalStartupCleanupEvidence(wireObject: JsonWireObject) throws -> WorkerPersistentPromptCacheStartupCleanupEvidence? {
        guard let evidenceObject = try wireObject.decodeOptionalObject(fieldName: "startup_cleanup_evidence") else {
            return nil;
        }
        return try WorkerPersistentPromptCacheStartupCleanupEvidence.fromWireValue(.object(evidenceObject));
    }

    private static func optionalUInt64WireValue(_ optionalValue: UInt64?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .unsignedInteger(unwrappedValue);
    }

    private static func optionalMissReasonWireValue(_ optionalValue: WorkerPersistentPromptCacheMissReason?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return unwrappedValue.wireValue();
    }

    private static func optionalBlockHashPrefixWireValue(_ optionalValue: WorkerPersistentPromptCacheExpectedBlockHashPrefix?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return unwrappedValue.wireValue();
    }

    private static func optionalStartupCleanupEvidenceWireValue(_ optionalValue: WorkerPersistentPromptCacheStartupCleanupEvidence?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return unwrappedValue.wireValue();
    }
}
