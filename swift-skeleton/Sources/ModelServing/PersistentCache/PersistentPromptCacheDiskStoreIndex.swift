import Foundation;

import ModelServing;

/// Process-local index of prompt-cache files already validated against
/// disk, port of the Rust `PersistentPromptCacheDiskStoreIndex`. The index
/// accelerates lookup and exposes counters, but it is never durable
/// authority: publication updates it only after commit, startup rebuilds it
/// from disk, and read paths remove entries whose files disappeared
/// concurrently.
public struct PersistentPromptCacheDiskStoreIndex: Sendable {

    /// One tracked cache-owned file with its validated size.
    public struct TrackedFile: Equatable, Sendable {

        public let filePath: String;

        public let fileSizeBytes: UInt64;

        public init(filePath: String, fileSizeBytes: UInt64) {
            self.filePath = filePath;
            self.fileSizeBytes = fileSizeBytes;
        }
    }

    /// One entry represents the directory as a topology unit. State files
    /// remain optional because retention may remove a redundant parent
    /// boundary while preserving its required sequence state and ancestry
    /// metadata.
    public struct TrackedBlock: Equatable, Sendable {

        public let blockDirectoryPath: String;

        public let blockIndex: UInt32;

        public let parentBlockHash: Data?;

        public let sequenceStateFile: TrackedFile?;

        public let boundaryStateFile: TrackedFile?;

        public init(
            blockDirectoryPath: String,
            blockIndex: UInt32,
            parentBlockHash: Data?,
            sequenceStateFile: TrackedFile?,
            boundaryStateFile: TrackedFile?
        ) {
            self.blockDirectoryPath = blockDirectoryPath;
            self.blockIndex = blockIndex;
            self.parentBlockHash = parentBlockHash;
            self.sequenceStateFile = sequenceStateFile;
            self.boundaryStateFile = boundaryStateFile;
        }
    }

    private var blocksByHash: [Data: TrackedBlock];

    private var visualEmbeddingsByHash: [Data: TrackedFile];

    public init() {
        self.blocksByHash = [:];
        self.visualEmbeddingsByHash = [:];
    }

    /// Tracks one flat cache-owned file under the given hash.
    mutating func insertVisualEmbeddingFile(fileHash: Data, trackedFile: TrackedFile) {
        self.visualEmbeddingsByHash[fileHash] = trackedFile;
    }

    /// Blocks whose sequence-state file is present and validated.
    public var sequenceStateBlockCount: Int {
        return self.blocksByHash.values
            .filter({ (trackedBlock: TrackedBlock) -> Bool in
                return trackedBlock.sequenceStateFile != nil;
            })
            .count;
    }

    /// Blocks whose boundary-state snapshot is present and validated.
    public var boundaryStateSnapshotCount: Int {
        return self.blocksByHash.values
            .filter({ (trackedBlock: TrackedBlock) -> Bool in
                return trackedBlock.boundaryStateFile != nil;
            })
            .count;
    }

    /// Persisted visual-embedding files tracked under the global root.
    public var visualEmbeddingCount: Int {
        return self.visualEmbeddingsByHash.count;
    }

    /// Returns one tracked block by its content hash.
    public func block(blockHash: Data) -> TrackedBlock? {
        return self.blocksByHash[blockHash];
    }

    /// Inserts one tracked block after its directory validated.
    public mutating func insertBlock(blockHash: Data, trackedBlock: TrackedBlock) {
        self.blocksByHash[blockHash] = trackedBlock;
    }

    /// Every tracked block with its content hash, in unspecified order.
    public func trackedBlocks() -> [(blockHash: Data, trackedBlock: TrackedBlock)] {
        return self.blocksByHash.map({ (blockHash: Data, trackedBlock: TrackedBlock) in
            return (blockHash, trackedBlock);
        });
    }

    /// Removes one tracked block, returning it when present.
    public mutating func removeBlock(blockHash: Data) -> TrackedBlock? {
        return self.blocksByHash.removeValue(forKey: blockHash);
    }

    /// Walks from tip to root collecting the directory paths quota eviction
    /// must protect, stopping on missing or cyclic topology. Startup
    /// validation should have removed both, but quota protection must
    /// remain bounded even if disk changes after the index was built.
    public func protectedAncestryDirectoryPaths(chainTipBlockHash: Data) -> [String] {
        var protectedBlockDirectoryPaths: [String] = [];
        var visitedBlockHashes: Set<Data> = [];
        var nextBlockHash: Data? = chainTipBlockHash;
        while let blockHash: Data = nextBlockHash {
            if visitedBlockHashes.contains(blockHash) {
                break;
            }
            visitedBlockHashes.insert(blockHash);
            guard let trackedBlock: TrackedBlock = self.blocksByHash[blockHash] else {
                break;
            }
            protectedBlockDirectoryPaths.append(trackedBlock.blockDirectoryPath);
            nextBlockHash = trackedBlock.parentBlockHash;
        }
        return protectedBlockDirectoryPaths;
    }

    /// Returns one tracked file by kind and content hash.
    public func file(
        sequenceState: Bool, fileHash: Data
    ) -> TrackedFile? {
        guard let trackedBlock: TrackedBlock = self.blocksByHash[fileHash] else {
            return nil;
        }
        return sequenceState ? trackedBlock.sequenceStateFile : trackedBlock.boundaryStateFile;
    }

    /// Inserts one tracked visual-embedding file.
    public mutating func insertVisualEmbedding(fileHash: Data, trackedFile: TrackedFile) {
        self.visualEmbeddingsByHash[fileHash] = trackedFile;
    }

    /// Removes one tracked visual-embedding file, returning it when present.
    public mutating func removeVisualEmbedding(fileHash: Data) -> TrackedFile? {
        return self.visualEmbeddingsByHash.removeValue(forKey: fileHash);
    }
}
