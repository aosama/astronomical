import Foundation;


/// Durable identity and topology record for one committed prompt-cache
/// block, port of the Rust `PersistentPromptCacheBlockManifest`. The
/// directory name proves only the block's content hash: this manifest also
/// binds that hash to its ordinal position, parent, model storage geometry,
/// and required state-file kinds. Readers validate all of those fields
/// before treating files in the directory as one link in a restorable chain.
public struct PersistentPromptCacheBlockManifest: Equatable, Sendable {

    private let formatVersionValue: String;
    private let blockHashHex: String;
    private let blockIndexValue: UInt32;
    private let parentBlockHashHex: String?;
    private let storageContractFingerprintValue: String;
    private let hasSequenceStateValue: Bool;
    private let hasBoundaryStateValue: Bool;

    private init(
        formatVersionValue: String,
        blockHashHex: String,
        blockIndexValue: UInt32,
        parentBlockHashHex: String?,
        storageContractFingerprintValue: String,
        hasSequenceStateValue: Bool,
        hasBoundaryStateValue: Bool
    ) {
        self.formatVersionValue = formatVersionValue;
        self.blockHashHex = blockHashHex;
        self.blockIndexValue = blockIndexValue;
        self.parentBlockHashHex = parentBlockHashHex;
        self.storageContractFingerprintValue = storageContractFingerprintValue;
        self.hasSequenceStateValue = hasSequenceStateValue;
        self.hasBoundaryStateValue = hasBoundaryStateValue;
    }

    /// Builds the manifest one publication stamps beside its state files.
    init(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        modelContract: PersistentPromptCacheModelContract
    ) {
        // These fields intentionally duplicate facts available elsewhere.
        // Keeping the complete contract in each block makes startup
        // validation local and prevents directory layout or file presence
        // from becoming implicit truth.
        self.formatVersionValue = PersistentPromptCacheBlockHeader.FORMAT_VERSION;
        self.blockHashHex = PersistentPromptCacheStoreFile.hexEncode(blockKey.blockHash());
        self.blockIndexValue = blockKey.blockIndex();
        self.parentBlockHashHex = parentBlockKey.map({ (parentBlockKey: PersistentPromptCacheBlockKey) -> String in
            return PersistentPromptCacheStoreFile.hexEncode(parentBlockKey.blockHash());
        });
        self.storageContractFingerprintValue = modelContract.storageContractFingerprintHex();
        self.hasSequenceStateValue = modelContract.hasSequenceState;
        self.hasBoundaryStateValue = modelContract.hasBoundaryState;
    }

    /// Reads and validates one committed block's manifest.
    public static func readFromBlockDirectory(
        blockDirectory: URL,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheBlockManifest {
        let unvalidatedManifest: PersistentPromptCacheBlockManifest = try
            PersistentPromptCacheBlockManifest.readUnvalidatedFromBlockDirectory(
                blockDirectory: blockDirectory);
        let manifestFilePath: URL = blockDirectory
            .appendingPathComponent(PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME);
        try unvalidatedManifest.validate(
            manifestFilePath: manifestFilePath, modelContract: modelContract);
        return unvalidatedManifest;
    }

    /// Reads one block manifest without applying the active-contract
    /// validation; startup scans use this before deciding which rejection
    /// to surface.
    public static func readUnvalidatedFromBlockDirectory(
        blockDirectory: URL
    ) throws -> PersistentPromptCacheBlockManifest {
        let manifestFilePath: URL = blockDirectory
            .appendingPathComponent(PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME);
        let manifestText: String;
        do {
            manifestText = try String(contentsOf: manifestFilePath, encoding: .utf8);
        } catch {
            throw PersistentPromptCacheDiskStoreError.readBlockManifest(
                manifestFilePath: manifestFilePath.path,
                problem: String(describing: error));
        }
        guard let manifestData: Data = manifestText.data(using: .utf8),
            let manifestObject: [String: Any] = try? JSONSerialization.jsonObject(
                with: manifestData, options: []) as? [String: Any],
            let formatVersion: String = manifestObject["format_version"] as? String,
            let blockHash: String = manifestObject["block_hash"] as? String,
            let blockIndex: UInt32 = (manifestObject["block_index"] as? NSNumber)?.uint32Value,
            let storageContractFingerprint: String =
                manifestObject["storage_contract_fingerprint"] as? String,
            let hasSequenceState: Bool = (manifestObject["has_sequence_state"] as? NSNumber)?.boolValue,
            let hasBoundaryState: Bool = (manifestObject["has_boundary_state"] as? NSNumber)?.boolValue
        else {
            throw PersistentPromptCacheDiskStoreError.parseBlockManifest(
                manifestFilePath: manifestFilePath.path,
                problem: "the manifest is not a complete block manifest document");
        }
        let parentBlockHash: String? = manifestObject["parent_block_hash"] as? String;
        return PersistentPromptCacheBlockManifest(
            formatVersionValue: formatVersion,
            blockHashHex: blockHash,
            blockIndexValue: blockIndex,
            parentBlockHashHex: parentBlockHash,
            storageContractFingerprintValue: storageContractFingerprint,
            hasSequenceStateValue: hasSequenceState,
            hasBoundaryStateValue: hasBoundaryState);
    }

    /// Commits the manifest inside the private staging directory: the
    /// temporary file is synchronized, then renamed over the final name, so
    /// the enclosing block directory publishes only after this record is
    /// durable.
    func writeToStagingDirectory(
        stagingBlockDirectory: URL
    ) throws -> URL {
        let manifestFilePath: URL = stagingBlockDirectory
            .appendingPathComponent(PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME);
        let temporaryManifestFilePath: URL = stagingBlockDirectory
            .appendingPathComponent("manifest.json.tmp");
        try PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
            filePath: temporaryManifestFilePath);
        let manifestDocument: [String: Any] = [
            "block_hash": self.blockHashHex,
            "block_index": self.blockIndexValue,
            "format_version": self.formatVersionValue,
            "has_boundary_state": self.hasBoundaryStateValue,
            "has_sequence_state": self.hasSequenceStateValue,
            // JSONSerialization cannot box a nil Optional; the absent-parent
            // case serializes as JSON null exactly like the Rust serde form.
            "parent_block_hash": self.parentBlockHashHex ?? NSNull(),
            "storage_contract_fingerprint": self.storageContractFingerprintValue,
        ];
        let manifestBytes: Data;
        do {
            manifestBytes = try JSONSerialization.data(
                withJSONObject: manifestDocument, options: [.sortedKeys]);
        } catch {
            throw PersistentPromptCacheDiskStoreError.serializeBlockManifest(
                problem: String(describing: error));
        }
        let temporaryFileCreated: Bool = FileManager.default.createFile(
            atPath: temporaryManifestFilePath.path, contents: nil, attributes: nil);
        if temporaryFileCreated == false {
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: temporaryManifestFilePath.path,
                problem: "the temporary manifest file could not be created");
        }
        let temporaryFileHandle: FileHandle;
        do {
            temporaryFileHandle = try FileHandle(forWritingTo: temporaryManifestFilePath);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryManifestFilePath);
            throw PersistentPromptCacheDiskStoreError.openTempFile(
                tempFilePath: temporaryManifestFilePath.path,
                problem: String(describing: error));
        }
        do {
            try temporaryFileHandle.write(contentsOf: manifestBytes);
            temporaryFileHandle.synchronizeFile();
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryManifestFilePath);
            throw PersistentPromptCacheDiskStoreError.writeTempFile(
                tempFilePath: temporaryManifestFilePath.path,
                problem: String(describing: error));
        }
        try? temporaryFileHandle.close();
        do {
            try FileManager.default.moveItem(
                at: temporaryManifestFilePath, to: manifestFilePath);
        } catch {
            try? PersistentPromptCacheStoreFile.removeCacheOwnedFileOrConfirmAbsent(
                filePath: temporaryManifestFilePath);
            throw PersistentPromptCacheDiskStoreError.renameTempFile(
                tempFilePath: temporaryManifestFilePath.path,
                blockFilePath: manifestFilePath.path,
                problem: String(describing: error));
        }
        return manifestFilePath;
    }

    /// The binary block hash this manifest binds.
    func blockHash() throws -> Data {
        guard let blockHash: Data = PersistentPromptCacheStoreFile.parseBlockHashHex(
            self.blockHashHex) else {
            throw PersistentPromptCacheDiskStoreError.invalidBlockManifest(
                manifestFilePath: PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME,
                description: "block hash is not a 32-byte lowercase hexadecimal value");
        }
        return blockHash;
    }

    /// The binary parent hash when the manifest binds one.
    func parentBlockHash() -> Data? {
        return self.parentBlockHashHex.flatMap({ (parentBlockHashHex: String) -> Data? in
            return PersistentPromptCacheStoreFile.parseBlockHashHex(parentBlockHashHex);
        });
    }

    /// The zero-based chain position this manifest binds.
    public var blockIndex: UInt32 {
        return self.blockIndexValue;
    }

    /// Whether the contract requires a sequence-state file in this block.
    public var hasSequenceState: Bool {
        return self.hasSequenceStateValue;
    }

    /// Whether the contract requires a boundary-state file in this block.
    public var hasBoundaryState: Bool {
        return self.hasBoundaryStateValue;
    }

    /// The storage-contract fingerprint stamped into the manifest.
    public var storageContractFingerprint: String {
        return self.storageContractFingerprintValue;
    }

    private func validate(
        manifestFilePath: URL,
        modelContract: PersistentPromptCacheModelContract
    ) throws {
        // Validation is deliberately fail-closed. A block from another
        // format, model revision, tensor layout, or state topology must
        // never be joined to the active request merely because its content
        // hash parses.
        if self.formatVersionValue != PersistentPromptCacheBlockHeader.FORMAT_VERSION {
            throw PersistentPromptCacheDiskStoreError.invalidBlockManifest(
                manifestFilePath: manifestFilePath.path,
                description: "format version does not match the active prompt-cache format");
        }
        if self.storageContractFingerprintValue != modelContract.storageContractFingerprintHex() {
            throw PersistentPromptCacheDiskStoreError.invalidBlockManifest(
                manifestFilePath: manifestFilePath.path,
                description: "storage contract fingerprint does not match the active model");
        }
        if self.hasSequenceStateValue != modelContract.hasSequenceState
            || self.hasBoundaryStateValue != modelContract.hasBoundaryState {
            throw PersistentPromptCacheDiskStoreError.invalidBlockManifest(
                manifestFilePath: manifestFilePath.path,
                description: "state topology does not match the active model contract");
        }
        if PersistentPromptCacheStoreFile.parseBlockHashHex(self.blockHashHex) == nil
            || (self.parentBlockHashHex != nil
                && PersistentPromptCacheStoreFile.parseBlockHashHex(self.parentBlockHashHex!)
                    == nil) {
            throw PersistentPromptCacheDiskStoreError.invalidBlockManifest(
                manifestFilePath: manifestFilePath.path,
                description: "block ancestry contains an invalid hash");
        }
    }
}
