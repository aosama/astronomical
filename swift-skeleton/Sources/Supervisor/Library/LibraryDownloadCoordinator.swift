import Foundation

import AstronomicalConfig
import CryptoKit
import HuggingFace

/// Thread-safe registry of identities this instance published and verified;
/// readable synchronously from the REST catalog projection.
public final class LibraryValidatedPublicationRegistry: @unchecked Sendable {
    private let stateLock: NSLock = NSLock()
    private var publications: Set<String> = []

    public init() {
    }

    public func snapshot() -> Set<String> {
        self.stateLock.lock()
        let currentPublications: Set<String> = self.publications
        self.stateLock.unlock()
        return currentPublications
    }

    public func insert(_ huggingfaceId: String) {
        self.stateLock.lock()
        self.publications.insert(huggingfaceId)
        self.stateLock.unlock()
    }
}

/**
 * Owns the single Library download this instance runs (#1059 shape): the
 * durable record holds lifecycle transitions only, the hub blob cache holds
 * partial bytes, and live per-file progress stays in memory. Publication
 * writes provenance, refreshes discovery, and only then releases the job.
 */
public actor LibraryDownloadCoordinator {

    /// Live progress the REST surface reads between durable transitions.
    public struct LiveProgress: Sendable {
        public let bytesCompleted: UInt64
        public let bytesTotal: UInt64
        public let currentFileRelativePath: String?
    }

    public enum CoordinatorError: Error, Equatable {
        case libraryBusy
        case catalogEntryNotFound
        case jobNotFound
    }

    private let downloadCatalog: DownloadCatalog
    private let modelsDirectory: FilePath
    private let stagingRootDirectory: FilePath
    private let hubDownloadService: HubDownloadService
    private let jobRecordStore: LibraryDownloadJobRecordStore
    private let discoveryRefresh: @Sendable () async throws -> Void
    private let diskPreflight: DownloadDiskPreflight<LibraryDownloadCoordinator.CoordinatorDiskCapacityQuery>

    private var activeTask: Task<Void, Never>?
    private var liveProgress: LiveProgress?
    public nonisolated let validatedPublications: LibraryValidatedPublicationRegistry = LibraryValidatedPublicationRegistry()

    public init(
        downloadCatalog: DownloadCatalog,
        modelsDirectory: FilePath,
        stateDirectory: FilePath,
        hubEndpoint: URL,
        discoveryRefresh: @escaping @Sendable () async throws -> Void,
        availableCapacityBytes: @escaping @Sendable (_ volumeDirectory: FilePath) -> UInt64?
    ) {
        self.downloadCatalog = downloadCatalog
        self.modelsDirectory = modelsDirectory
        self.stagingRootDirectory = stateDirectory.appending(component: "download-staging")
        self.hubDownloadService = HubDownloadService(
            hubEndpoint: hubEndpoint,
            cacheDirectory: stateDirectory.appending(component: "hub-blob-cache"))
        self.jobRecordStore = LibraryDownloadJobRecordStore(stateDirectory: stateDirectory)
        self.discoveryRefresh = discoveryRefresh
        self.diskPreflight = DownloadDiskPreflight(
            capacityQuery: LibraryDownloadCoordinator.CoordinatorDiskCapacityQuery(
                availableCapacityBytes: availableCapacityBytes))
    }

    /// Adapts the coordinator's injectable volume-capacity closure onto the
    /// typed preflight query seam; an unanswered query is a typed failure,
    /// not a silently skipped check.
    private struct CoordinatorDiskCapacityQuery: DiskCapacityQuery {

        private let availableCapacityBytes: @Sendable (_ volumeDirectory: FilePath) -> UInt64?;

        fileprivate init(
            availableCapacityBytes: @escaping @Sendable (_ volumeDirectory: FilePath) -> UInt64?
        ) {
            self.availableCapacityBytes = availableCapacityBytes;
        }

        func availableSpaceBytes(existingSameVolumePath: FilePath) -> Result<UInt64, DiskCapacityQueryError> {
            if let availableBytes: UInt64 = self.availableCapacityBytes(existingSameVolumePath) {
                return .success(availableBytes);
            }
            return .failure(.volumeUnavailable(
                reason: "the volume capacity query returned no answer"));
        }
    }

    public func recoverStartupState() {
        self.recoverValidatedPublications()
        guard let record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            return
        }
        if record.state == .publishing {
            self.spawnPublicationTask(record: record)
            return
        }
        if record.state.isInterruptedByRestart {
            var pausedRecord: LibraryDownloadJobRecord = record
            pausedRecord.state = .paused
            pausedRecord.errorCode = nil
            try? self.jobRecordStore.save(pausedRecord)
        }
    }

    /// The current job with live bytes merged in while a transfer runs.
    public func currentJob() -> LibraryDownloadJobRecord? {
        guard var record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            return nil
        }
        if let liveProgress: LiveProgress = self.liveProgress {
            record.bytesCompleted = liveProgress.bytesCompleted
            record.bytesTotal = liveProgress.bytesTotal
        }
        return record
    }

    public nonisolated func destinationDirectory(huggingfaceId: String) -> FilePath {
        return LibraryDownloadPublication.destinationDirectory(
            huggingfaceId: huggingfaceId,
            modelsDirectory: self.modelsDirectory)
    }

    public func start(huggingfaceId: String) throws {
        if self.activeTask != nil || (try? self.jobRecordStore.load()) != nil {
            throw CoordinatorError.libraryBusy
        }
        guard let catalogEntry: DownloadCatalogEntry = self.downloadCatalog.entries.first(
            where: { (entry: DownloadCatalogEntry) -> Bool in
                return entry.huggingfaceId == huggingfaceId
            })
        else {
            throw CoordinatorError.catalogEntryNotFound
        }
        self.spawnDownloadTask(catalogEntry: catalogEntry, resumeHint: nil)
    }

    public func resume() throws {
        if self.activeTask != nil {
            throw CoordinatorError.libraryBusy
        }
        guard let record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            throw CoordinatorError.jobNotFound
        }
        if record.state == .publishing {
            self.spawnPublicationTask(record: record)
            return
        }
        guard record.state == .paused || record.state == .failed else {
            throw CoordinatorError.libraryBusy
        }
        guard let catalogEntry: DownloadCatalogEntry = self.downloadCatalog.entries.first(
            where: { (entry: DownloadCatalogEntry) -> Bool in
                return entry.huggingfaceId == record.huggingfaceId
                    && entry.revision == record.revision
            })
        else {
            throw CoordinatorError.catalogEntryNotFound
        }
        self.spawnDownloadTask(catalogEntry: catalogEntry, resumeHint: record)
    }

    public func pause() throws -> LibraryDownloadJobRecord {
        self.cancelActiveTask()
        guard var record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            throw CoordinatorError.jobNotFound
        }
        record.state = .paused
        record.errorCode = nil
        try? self.jobRecordStore.save(record)
        return record
    }

    public func cancel() {
        self.cancelActiveTask()
        self.hubDownloadService.removeIncompleteBlobs(repositoryId: self.lastRecordRepositoryId() ?? "")
        try? FileManager.default.removeItem(atPath: self.stagingRootDirectory.string)
        self.jobRecordStore.delete()
        self.liveProgress = nil
    }

    // MARK: - Task bodies

    private func spawnDownloadTask(
        catalogEntry: DownloadCatalogEntry,
        resumeHint: LibraryDownloadJobRecord?
    ) {
        let startingState: LibraryDownloadJobRecordState = resumeHint != nil ? .downloading : .checkingDisk
        let startingBytesTotal: UInt64 = resumeHint?.bytesTotal ?? catalogEntry.approximateSizeBytes
        self.persistState(
            huggingfaceId: catalogEntry.huggingfaceId,
            revision: catalogEntry.revision,
            state: startingState,
            bytesCompleted: 0,
            bytesTotal: startingBytesTotal,
            errorCode: nil)
        self.activeTask = Task<Void, Never> { [weak self] in
            await self?.runDownload(catalogEntry: catalogEntry)
            await self?.finishTaskSlot()
        }
    }

    private func runDownload(catalogEntry: DownloadCatalogEntry) async {
        let identity: String = catalogEntry.huggingfaceId
        let destination: FilePath = LibraryDownloadPublication.destinationDirectory(
            huggingfaceId: identity,
            modelsDirectory: self.modelsDirectory)
        do {
            do {
                _ = try self.diskPreflight.checkInitialDownload(
                    existingSameVolumePath: self.modelsDirectory,
                    catalogApproximateBytes: catalogEntry.approximateSizeBytes);
            } catch {
                self.failJob(errorCode: .insufficientDisk);
                return;
            }
            _ = try await self.hubDownloadService.fetchModelMetadata(
                repositoryId: identity,
                revision: catalogEntry.revision)
            if self.hasMatchingProvenance(catalogEntry: catalogEntry, destination: destination) {
                await self.publish(catalogEntry: catalogEntry, destination: destination)
                return
            }
            self.transition(.fetchingManifest)
            let repositoryFiles: [HubRepositoryFile] = try await self.hubDownloadService.listRepositoryFiles(
                repositoryId: identity,
                revision: catalogEntry.revision)
            if FileManager.default.fileExists(atPath: destination.string) {
                // A pre-existing destination adopts only when every manifest
                // file is byte-identical; anything else is a foreign model
                // that must not be touched.
                let isExactPublication: Bool = self.matchesManifestExactly(
                    repositoryFiles: repositoryFiles,
                    destination: destination)
                if isExactPublication {
                    await self.publish(catalogEntry: catalogEntry, destination: destination)
                } else {
                    self.failJob(errorCode: .modelAlreadyPresent)
                }
                return
            }
            let exactBytesTotal: UInt64 = repositoryFiles.reduce(UInt64(0)) { (partial: UInt64, repositoryFile: HubRepositoryFile) -> UInt64 in
                return partial + (repositoryFile.sizeBytes ?? 0)
            }
            self.transition(.downloading, bytesTotal: exactBytesTotal)
            try? FileManager.default.removeItem(atPath: self.stagingRootDirectory.string)
            let stagingDestination: FilePath = self.stagingRootDirectory
                .appending(component: catalogEntry.huggingfaceId)
            let destinationUrl: URL = URL(fileURLWithPath: stagingDestination.string)
            _ = try await self.hubDownloadService.downloadSnapshot(
                repositoryId: identity,
                revision: catalogEntry.revision,
                matching: [],
                destinationDirectory: destinationUrl,
                progressRows: { [weak self] (fileRows: [SnapshotFileProgress]) in
                    guard let self: LibraryDownloadCoordinator = self else { return }
                    Task<Void, Never> {
                        await self.recordLiveProgress(fileRows)
                    }
                })
            self.transition(.verifying)
            try await self.verifyDownloadedFiles(
                repositoryId: identity,
                repositoryFiles: repositoryFiles,
                destination: stagingDestination)
            await self.publish(catalogEntry: catalogEntry, destination: destination, stagingDestination: stagingDestination)
        } catch {
            // Pause and cancel surface as task cancellation — the durable
            // record must never turn that into a failure.
            if Task.isCancelled {
                self.transitionIfRecorded(.paused)
            } else if let hubError: HubDownloadError = error as? HubDownloadError {
                let errorCode: LibraryDownloadPublicErrorCode
                switch hubError {
                case .gated:
                    errorCode = .downloadGated
                case .checksumMismatch:
                    errorCode = .checksumMismatch
                default:
                    errorCode = .downloadFailed
                }
                self.failJob(errorCode: errorCode)
            } else {
                self.failJob(errorCode: .downloadFailed)
            }
        }
    }

    private func verifyDownloadedFiles(
        repositoryId: String,
        repositoryFiles: [HubRepositoryFile],
        destination: FilePath
    ) async throws {
        for repositoryFile: HubRepositoryFile in repositoryFiles {
            let filePath: String = destination.string + "/" + repositoryFile.relativePath
            guard let fileBytes: Data = FileManager.default.contents(atPath: filePath) else {
                throw HubDownloadError.transport("published file missing")
            }
            if let expectedSha256: String = repositoryFile.sha256Digest {
                let actualHex: String = SHA256.hash(data: fileBytes).map({ (digestByte: UInt8) -> String in
                    return String(format: "%02x", digestByte)
                }).joined()
                if actualHex != expectedSha256 {
                    // The blob the cache believes complete is corrupt —
                    // purge the whole repository entry so a retry must
                    // redownload every byte, and drop the staged copy.
                    self.hubDownloadService.removeCachedRepository(repositoryId: repositoryId)
                    try? FileManager.default.removeItem(atPath: destination.string)
                    throw HubDownloadError.checksumMismatch
                }
            }
        }
    }

    private func publish(
        catalogEntry: DownloadCatalogEntry,
        destination: FilePath,
        stagingDestination: FilePath? = nil
    ) async {
        self.transition(.publishing)
        do {
            if let stagingDestination: FilePath = stagingDestination {
                try FileManager.default.createDirectory(
                    atPath: destination.parentDirectory()?.string ?? self.modelsDirectory.string,
                    withIntermediateDirectories: true)
                if FileManager.default.fileExists(atPath: destination.string) {
                    try FileManager.default.removeItem(atPath: destination.string)
                }
                try FileManager.default.moveItem(
                    at: URL(fileURLWithPath: stagingDestination.string),
                    to: URL(fileURLWithPath: destination.string))
                try? FileManager.default.removeItem(atPath: self.stagingRootDirectory.string)
            }
            if LibraryDownloadPublication.existingProvenance(destinationDirectory: destination) == nil {
                try LibraryDownloadPublication.writeProvenance(
                    destinationDirectory: destination,
                    providerModelId: catalogEntry.huggingfaceId,
                    revision: catalogEntry.revision)
            }
            try await self.discoveryRefresh()
            self.validatedPublications.insert(catalogEntry.huggingfaceId)
            self.jobRecordStore.delete()
            self.liveProgress = nil
            self.hubDownloadService.removeCachedRepository(repositoryId: catalogEntry.huggingfaceId)
        } catch {
            // A failed discovery refresh strands the job in publishing —
            // the exact stuck state journey 4 pins.
        }
    }

    private func spawnPublicationTask(record: LibraryDownloadJobRecord) {
        self.activeTask = Task<Void, Never> { [weak self] in
            guard let self: LibraryDownloadCoordinator = self else { return }
            guard let catalogEntry: DownloadCatalogEntry = self.downloadCatalog.entries.first(
                where: { (entry: DownloadCatalogEntry) -> Bool in
                    return entry.huggingfaceId == record.huggingfaceId
                        && entry.revision == record.revision
                })
            else {
                return
            }
            let destination: FilePath = LibraryDownloadPublication.destinationDirectory(
                huggingfaceId: record.huggingfaceId,
                modelsDirectory: self.modelsDirectory)
            await self.publish(catalogEntry: catalogEntry, destination: destination)
            await self.finishTaskSlot()
        }
    }

    private func hasMatchingProvenance(
        catalogEntry: DownloadCatalogEntry,
        destination: FilePath
    ) -> Bool {
        guard let provenance: (providerModelId: String, revision: String) =
            LibraryDownloadPublication.existingProvenance(destinationDirectory: destination)
        else {
            return false
        }
        return provenance.providerModelId == catalogEntry.huggingfaceId
            && provenance.revision == catalogEntry.revision
    }

    /// Byte-equality of the destination against the manifest: an LFS file
    /// verifies by SHA-256 against the tree digest, a plain Git file by its
    /// blob SHA-1.
    private func matchesManifestExactly(
        repositoryFiles: [HubRepositoryFile],
        destination: FilePath
    ) -> Bool {
        for repositoryFile: HubRepositoryFile in repositoryFiles {
            let filePath: String = destination.string + "/" + repositoryFile.relativePath
            guard let fileBytes: Data = FileManager.default.contents(atPath: filePath) else {
                return false
            }
            if let expectedSha256: String = repositoryFile.sha256Digest {
                if LibraryDownloadFileDigest.sha256Hex(fileBytes) != expectedSha256 {
                    return false
                }
            } else if let expectedBytes: UInt64 = repositoryFile.sizeBytes {
                if UInt64(fileBytes.count) != expectedBytes {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Record plumbing

    private func recordLiveProgress(_ fileRows: [SnapshotFileProgress]) {
        var bytesCompleted: UInt64 = 0
        var bytesTotal: UInt64 = 0
        var currentFileRelativePath: String? = nil
        for fileRow: SnapshotFileProgress in fileRows {
            let rowTotal: UInt64 = UInt64(fileRow.sizeBytes ?? 0)
            bytesTotal = bytesTotal + rowTotal
            bytesCompleted = bytesCompleted + UInt64((Double(rowTotal) * fileRow.fractionCompleted).rounded())
            if fileRow.fractionCompleted < 1.0 && currentFileRelativePath == nil {
                currentFileRelativePath = fileRow.path
            }
        }
        self.liveProgress = LiveProgress(
            bytesCompleted: bytesCompleted,
            bytesTotal: bytesTotal,
            currentFileRelativePath: currentFileRelativePath)
    }

    private func transition(
        _ state: LibraryDownloadJobRecordState,
        bytesTotal: UInt64? = nil
    ) {
        self.transitionIfRecorded(state, bytesTotal: bytesTotal)
    }

    private func transitionIfRecorded(
        _ state: LibraryDownloadJobRecordState,
        bytesTotal: UInt64? = nil
    ) {
        guard var record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            return
        }
        record.state = state
        record.errorCode = nil
        if let bytesTotal: UInt64 = bytesTotal {
            record.bytesTotal = bytesTotal
        }
        try? self.jobRecordStore.save(record)
    }

    private func failJob(errorCode: LibraryDownloadPublicErrorCode) {
        guard var record: LibraryDownloadJobRecord = try? self.jobRecordStore.load() else {
            return
        }
        record.state = .failed
        record.errorCode = errorCode.rawValue
        try? self.jobRecordStore.save(record)
        self.liveProgress = nil
    }

    private func persistState(
        huggingfaceId: String,
        revision: String,
        state: LibraryDownloadJobRecordState,
        bytesCompleted: UInt64,
        bytesTotal: UInt64,
        errorCode: String?
    ) {
        let record: LibraryDownloadJobRecord = LibraryDownloadJobRecord(
            huggingfaceId: huggingfaceId,
            revision: revision,
            state: state,
            bytesCompleted: bytesCompleted,
            bytesTotal: bytesTotal,
            errorCode: errorCode,
            updatedAtUnixMillis: UInt64(Date().timeIntervalSince1970 * 1000))
        try? self.jobRecordStore.save(record)
        self.liveProgress = nil
    }

    /// A destination carrying our provenance for a catalog identity was
    /// verified when it was written — readiness survives restarts.
    private func recoverValidatedPublications() {
        for catalogEntry: DownloadCatalogEntry in self.downloadCatalog.entries {
            let destination: FilePath = LibraryDownloadPublication.destinationDirectory(
                huggingfaceId: catalogEntry.huggingfaceId,
                modelsDirectory: self.modelsDirectory)
            if self.hasMatchingProvenance(catalogEntry: catalogEntry, destination: destination) {
                self.validatedPublications.insert(catalogEntry.huggingfaceId)
            }
        }
    }

    private func finishTaskSlot() {
        self.activeTask = nil
    }

    private func cancelActiveTask() {
        self.activeTask?.cancel()
        self.activeTask = nil
        self.liveProgress = nil
    }

    private func lastRecordRepositoryId() -> String? {
        return (try? self.jobRecordStore.load())?.huggingfaceId
    }
}
