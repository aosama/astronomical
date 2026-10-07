import Foundation

import AstronomicalConfig
import HuggingFace

/**
 * The one seam over the swift-huggingface package (#1059). Everything the
 * Library coordinator knows about the Hub flows through here: metadata
 * preflight, snapshot download with per-file progress, and cache lifecycle.
 * The custom `hubEndpoint` is what lets hermetic journeys run the real
 * package client against a scripted local hub.
 */
public final class HubDownloadService: @unchecked Sendable {

    public static let productionHubEndpoint: URL = URL(string: "https://huggingface.co")!

    private let hubClient: HubClient
    private let hubCache: HubCache?

    public init(hubEndpoint: URL, cacheDirectory: FilePath?) {
        self.hubCache = cacheDirectory.map { (cacheDirectory: FilePath) -> HubCache in
            return HubCache(cacheDirectory: URL(fileURLWithPath: cacheDirectory.string))
        }
        self.hubClient = HubClient(
            host: hubEndpoint,
            tokenProvider: .none,
            cache: self.hubCache)
    }

    /// Public repository metadata for the coordinator's admission preflight.
    public func fetchModelMetadata(
        repositoryId: String,
        revision: String?
    ) async throws -> HubModelMetadata {
        let repositoryIdentifier: Repo.ID = try HubDownloadService.repositoryIdentifier(repositoryId)
        do {
            let model: Model = try await self.hubClient.getModel(repositoryIdentifier, revision: revision)
            if model.gated == .auto || model.gated == .manual {
                throw HubDownloadError.gated
            }
            if model.visibility == .private {
                throw HubDownloadError.notPublic
            }
            return HubModelMetadata(commitHash: model.sha)
        } catch let error as HubDownloadError {
            throw error
        } catch let HTTPClientError.responseError(response, _) {
            throw HubDownloadService.statusError(response.statusCode)
        } catch {
            throw HubDownloadError.transport(String(describing: error))
        }
    }

    /**
     * Downloads one snapshot into the destination directory through the
     * package's ETag blob cache, so partial bytes resume across restarts.
     * `progressRows` receives the sampled per-file rows; bytes observed this
     * way are live-only — the caller decides what becomes durable.
     */
    public func downloadSnapshot(
        repositoryId: String,
        revision: String,
        matching: Array<String>,
        destinationDirectory: URL,
        progressRows: @escaping @Sendable (Array<SnapshotFileProgress>) -> Void
    ) async throws -> URL {
        let repositoryIdentifier: Repo.ID = try HubDownloadService.repositoryIdentifier(repositoryId)
        do {
            return try await self.hubClient.downloadSnapshot(
                of: repositoryIdentifier,
                kind: .model,
                to: destinationDirectory,
                revision: revision,
                matching: matching,
                progressHandler: nil,
                fileProgressHandler: { (fileRows: Array<SnapshotFileProgress>) in
                    progressRows(fileRows)
                })
        } catch let HTTPClientError.responseError(response, _) {
            throw HubDownloadService.statusError(response.statusCode)
        } catch {
            throw HubDownloadError.transport(String(describing: error))
        }
    }

    /// Lists the repository tree, exposing the per-file digests the checksum
    /// verifier needs. Entries are restricted to regular files.
    public func listRepositoryFiles(
        repositoryId: String,
        revision: String
    ) async throws -> Array<HubRepositoryFile> {
        let repositoryIdentifier: Repo.ID = try HubDownloadService.repositoryIdentifier(repositoryId)
        do {
            let treePages: Pages<Git.TreeEntry> = try await self.hubClient.listAllTree(
                in: repositoryIdentifier,
                kind: .model,
                revision: revision,
                recursive: true)
            var treeEntries: Array<Git.TreeEntry> = Array()
            for try await page: PaginatedResponse<Git.TreeEntry> in treePages {
                treeEntries.append(contentsOf: page.items)
            }
            return treeEntries.compactMap({ (treeEntry: Git.TreeEntry) -> HubRepositoryFile? in
                guard treeEntry.type == .file else {
                    return nil
                }
                return HubRepositoryFile(
                    relativePath: treeEntry.path,
                    sizeBytes: treeEntry.effectiveSize.map({ (effectiveSize: Int) -> UInt64 in
                        return UInt64(effectiveSize)
                    }),
                    sha256Digest: treeEntry.lfs?.oid)
            })
        } catch let HTTPClientError.responseError(response, _) {
            throw HubDownloadService.statusError(response.statusCode)
        } catch {
            throw HubDownloadError.transport(String(describing: error))
        }
    }

    /// Removes the repository's whole cache entry after publication so steady
    /// state holds one copy of the model, under the models directory.
    public func removeCachedRepository(repositoryId: String) {
        guard let hubCache: HubCache = self.hubCache,
            let repositoryIdentifier: Repo.ID = Repo.ID(rawValue: repositoryId)
        else {
            return
        }
        try? FileManager.default.removeItem(at: hubCache.repoDirectory(
            repo: repositoryIdentifier,
            kind: .model))
    }

    /// Removes the repository's partial transfer blobs — the cancel path's
    /// "no staged bytes survive" outcome from #1059.
    public func removeIncompleteBlobs(repositoryId: String) {
        guard let hubCache: HubCache = self.hubCache,
            let repositoryIdentifier: Repo.ID = Repo.ID(rawValue: repositoryId)
        else {
            return
        }
        let blobsDirectory: URL = hubCache.blobsDirectory(repo: repositoryIdentifier, kind: .model)
        guard let incompleteBlobNames: Array<String> = try? FileManager.default
            .contentsOfDirectory(atPath: blobsDirectory.path)
            .filter({ (blobName: String) -> Bool in
                return blobName.contains(".incomplete")
            })
        else {
            return
        }
        for incompleteBlobName: String in incompleteBlobNames {
            try? FileManager.default.removeItem(
                at: blobsDirectory.appendingPathComponent(incompleteBlobName))
        }
    }

    private static func repositoryIdentifier(_ repositoryId: String) throws -> Repo.ID {
        guard let repositoryIdentifier: Repo.ID = Repo.ID(rawValue: repositoryId) else {
            throw HubDownloadError.invalidRepositoryId
        }
        return repositoryIdentifier
    }

    private static func statusError(_ statusCode: Int) -> HubDownloadError {
        if statusCode == 401 || statusCode == 403 {
            return .gated
        }
        return .unexpectedStatus(statusCode)
    }
}

/// The admission-relevant subset of upstream model metadata.
public struct HubModelMetadata: Sendable {
    public let commitHash: String?
}

/// One regular file in a repository tree, with its independently verifiable digest.
public struct HubRepositoryFile: Sendable {
    public let relativePath: String
    public let sizeBytes: UInt64?
    public let sha256Digest: String?
}

/// Hub failures the coordinator maps onto stable public error codes.
public enum HubDownloadError: Error, Equatable {
    case gated
    case notPublic
    case invalidRepositoryId
    case unexpectedStatus(Int)
    case checksumMismatch
    case transport(String)
}
