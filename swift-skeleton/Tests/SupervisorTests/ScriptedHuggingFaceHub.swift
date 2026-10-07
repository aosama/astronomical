import Foundation

import CryptoKit
import Network

/// Runs its action exactly once across concurrent handler invocations.
private final class OneShotResumeGuard: @unchecked Sendable {
    private let stateLock: NSLock = NSLock()
    private var alreadyRan: Bool = false

    func runIfNeeded(_ action: () -> Void) {
        self.stateLock.lock()
        let shouldRun: Bool = !self.alreadyRan
        if shouldRun {
            self.alreadyRan = true
        }
        self.stateLock.unlock()
        if shouldRun {
            action()
        }
    }
}

/**
 * A scripted local hub for hermetic Library journeys: a real HTTP server on
 * loopback speaking the Hugging Face wire subset the swift-huggingface
 * client calls — model metadata, the recursive tree listing, and resolve
 * downloads with ETag preflight, HTTP Range resume, and paced chunked
 * bodies so live progress is observable mid-transfer.
 */
final class ScriptedHuggingFaceHub: @unchecked Sendable {

    struct ScriptedHubFile: Sendable {
        let relativePath: String
        let bytes: Data
        let servesAsLfs: Bool

        init(relativePath: String, bytes: Data, servesAsLfs: Bool = true) {
            self.relativePath = relativePath
            self.bytes = bytes
            self.servesAsLfs = servesAsLfs
        }
    }

    struct ScriptedRepository: Sendable {
        let repositoryId: String
        let revision: String
        let gated: Bool
        let isPrivate: Bool
        let files: Array<ScriptedHubFile>
    }

    private let listener: NWListener
    private let queue: DispatchQueue = DispatchQueue(label: "scripted-huggingface-hub")
    private let stateQueue: DispatchQueue = DispatchQueue(label: "scripted-huggingface-hub-state")
    private var connections: Array<NWConnection> = Array()
    private let repositories: Array<ScriptedRepository>
    private let chunkByteCount: Int
    private let chunkDelayMilliseconds: Int
    private var payloadBytesStreamed: UInt64 = 0
    private var resolveRequestCount: Int = 0
    private var failResolveRequests: Bool = false
    private var fileByteOverrides: [String: Data] = [:]
    private var resolveByteOverrides: [String: Data] = [:]

    init(
        repositories: Array<ScriptedRepository>,
        chunkByteCount: Int = 64 * 1024,
        chunkDelayMilliseconds: Int = 15
    ) throws {
        self.repositories = repositories
        self.chunkByteCount = chunkByteCount
        self.chunkDelayMilliseconds = chunkDelayMilliseconds
        let parameters: NWParameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        self.listener = try NWListener(using: parameters)
    }

    func start() async throws -> URL {
        let resumeGuard: OneShotResumeGuard = OneShotResumeGuard()
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
            self.queue.sync {
                self.listener.stateUpdateHandler = { (listenerState: NWListener.State) in
                    switch listenerState {
                    case .ready:
                        resumeGuard.runIfNeeded {
                            let port: UInt16 = self.listener.port?.rawValue ?? 0
                            continuation.resume(
                                returning: URL(string: "http://127.0.0.1:\(port)")!)
                        }
                    case .failed(let listenerError):
                        resumeGuard.runIfNeeded {
                            continuation.resume(throwing: listenerError)
                        }
                    default:
                        break
                    }
                }
                self.listener.newConnectionHandler = { (connection: NWConnection) in
                    self.connections.append(connection)
                    connection.start(queue: self.queue)
                    self.receiveRequest(connection, accumulated: Data())
                }
                self.listener.start(queue: self.queue)
            }
        }
    }

    func stop() {
        self.queue.sync {
            self.listener.cancel()
            self.listener.stateUpdateHandler = nil
            self.listener.newConnectionHandler = nil
            for connection: NWConnection in self.connections {
                connection.cancel()
            }
            self.connections.removeAll()
        }
    }

    var streamedPayloadByteCount: UInt64 {
        return self.stateQueue.sync(execute: { return self.payloadBytesStreamed; })
    }

    var resolveRequestTotal: Int {
        return self.stateQueue.sync(execute: { return self.resolveRequestCount; })
    }

    func failAllResolveRequests() {
        self.stateQueue.sync { self.failResolveRequests = true }
    }

    /// Swaps the served (and digested) bytes of one file — the corrupt-then-
    /// fixed retry journey depends on the tree digest following the new bytes.
    func replaceFileBytes(repositoryId: String, relativePath: String, bytes: Data) {
        self.stateQueue.sync {
            self.fileByteOverrides["\(repositoryId)\n\(relativePath)"] = bytes
        }
    }

    /// Serves different bytes than the tree digest advertises — the wire-level
    /// corruption case; verification must catch it.
    func corruptServedBytes(repositoryId: String, relativePath: String, bytes: Data) {
        self.stateQueue.sync {
            self.resolveByteOverrides["\(repositoryId)\n\(relativePath)"] = bytes
        }
    }

    func clearServedByteCorruption(repositoryId: String, relativePath: String) {
        _ = self.stateQueue.sync {
            () -> Bool in
            return self.resolveByteOverrides.removeValue(
                forKey: "\(repositoryId)\n\(relativePath)") != nil
        }
    }

    private func servedBytes(of hubFile: ScriptedHubFile, repositoryId: String) -> Data {
        return self.stateQueue.sync(execute: {
            let overrideKey: String = "\(repositoryId)\n\(hubFile.relativePath)"
            return self.resolveByteOverrides[overrideKey]
                ?? self.fileByteOverrides[overrideKey]
                ?? hubFile.bytes
        })
    }

    private func effectiveBytes(of hubFile: ScriptedHubFile, repositoryId: String) -> Data {
        return self.stateQueue.sync(execute: {
            return self.fileByteOverrides["\(repositoryId)\n\(hubFile.relativePath)"] ?? hubFile.bytes
        })
    }

    // MARK: - Request loop

    private func receiveRequest(_ connection: NWConnection, accumulated: Data) {
        let unsafeConnection: NWConnection = connection
        unsafeConnection.receive(minimumIncompleteLength: 1, maximumLength: 256 * 1024) {
            (receivedData: Data?, _: NWConnection.ContentContext?, isComplete: Bool, receiveError: NWError?) in
            var requestData: Data = accumulated
            if let receivedData: Data = receivedData {
                requestData.append(receivedData)
            }
            guard receiveError == nil else {
                unsafeConnection.cancel()
                return
            }
            guard let headerEndRange: Range<Data.Index> = requestData.range(of: Data("\r\n\r\n".utf8)) else {
                if isComplete {
                    unsafeConnection.cancel()
                    return
                }
                self.receiveRequest(unsafeConnection, accumulated: requestData)
                return
            }
            guard let requestText: String = String(data: requestData[..<headerEndRange.upperBound], encoding: .utf8) else {
                unsafeConnection.cancel()
                return
            }
            let requestLines: Array<Substring> = requestText.split(separator: "\r\n")
            guard let requestLine: Substring = requestLines.first else {
                unsafeConnection.cancel()
                return
            }
            let requestTokens: Array<Substring> = requestLine.split(separator: " ")
            guard requestTokens.count >= 2 else {
                unsafeConnection.cancel()
                return
            }
            let method: String = String(requestTokens[0]).uppercased()
            let rawTarget: String = String(requestTokens[1])
            let path: String = String(rawTarget.split(separator: "?").first ?? Substring(rawTarget))
            var headerFields: Dictionary<String, String> = Dictionary()
            for headerLine: Substring in requestLines.dropFirst() {
                let headerTokens: Array<Substring> = headerLine.split(separator: ":", maxSplits: 1)
                guard headerTokens.count == 2 else {
                    continue
                }
                headerFields[String(headerTokens[0]).lowercased()] = String(headerTokens[1]).trimmingCharacters(in: .whitespaces)
            }
            self.serve(method: method, path: path, headers: headerFields, connection: connection)
        }
    }

    private func serve(
        method: String,
        path: String,
        headers: Dictionary<String, String>,
        connection: NWConnection
    ) {
        if path.hasPrefix("/api/models/") && path.contains("/tree/") {
            self.serveTreeListing(path: path, connection: connection)
        } else if path.hasPrefix("/api/models/") {
            self.serveModelMetadata(path: path, connection: connection)
        } else if let (repository, hubFile): (ScriptedRepository, ScriptedHubFile) = self.resolveRepositoryAndFilePath(path) {
            self.serveResolve(
                method: method,
                repository: repository,
                hubFile: hubFile,
                rangeHeader: headers["range"],
                connection: connection)
        } else {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 404 Not Found",
                headers: [:],
                body: Data())
        }
        self.receiveRequest(connection, accumulated: Data())
    }

    private func serveModelMetadata(path: String, connection: NWConnection) {
        var repositoryId: String = String(path.dropFirst("/api/models/".count))
        if let revisionRange: Range<String.Index> = repositoryId.range(of: "/revision/") {
            repositoryId = String(repositoryId[..<revisionRange.lowerBound])
        }
        guard let repository: ScriptedRepository = self.repository(repositoryId) else {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 404 Not Found",
                headers: [:],
                body: Data())
            return
        }
        if repository.gated {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 403 Forbidden",
                headers: [:],
                body: Data())
            return
        }
        let metadataJson: String = """
            {"id": "\(repository.repositoryId)", "sha": "\(repository.revision)", "private": \(repository.isPrivate), "gated": false}
            """
        self.writePlainResponse(
            connection,
            statusLine: "HTTP/1.1 200 OK",
            headers: ["Content-Type": "application/json"],
            body: Data(metadataJson.utf8))
    }

    private func serveTreeListing(path: String, connection: NWConnection) {
        guard let treeRange: Range<String.Index> = path.range(of: "/tree/") else {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 404 Not Found",
                headers: [:],
                body: Data())
            return
        }
        let repositoryId: String = String(
            path[path.index(path.startIndex, offsetBy: "/api/models/".count) ..< treeRange.lowerBound])
        let revision: String = String(path[treeRange.upperBound...].split(separator: "/").first ?? "")
        guard let repository: ScriptedRepository = self.repository(repositoryId),
            repository.revision == revision
        else {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 404 Not Found",
                headers: [:],
                body: Data())
            return
        }
        let entryDocuments: Array<String> = repository.files.map({ (hubFile: ScriptedHubFile) -> String in
            let effectiveBytes: Data = self.effectiveBytes(of: hubFile, repositoryId: repositoryId)
            let lfsBlock: String
            if hubFile.servesAsLfs {
                lfsBlock = ", \"lfs\": {\"oid\": \"\(ScriptedHuggingFaceHub.sha256Hex(effectiveBytes))\", \"size\": \(effectiveBytes.count), \"pointerSize\": 134}"
            } else {
                lfsBlock = ""
            }
            let entryJson: String = "{\"type\": \"file\", \"oid\": \"\(ScriptedHuggingFaceHub.gitBlobSha1Hex(effectiveBytes))\", \"size\": \(effectiveBytes.count), \"path\": \"\(hubFile.relativePath)\"\(lfsBlock)}"
            return entryJson
        })
        let treeJson: String = "[\n" + entryDocuments.joined(separator: ",\n") + "\n]"
        self.writePlainResponse(
            connection,
            statusLine: "HTTP/1.1 200 OK",
            headers: ["Content-Type": "application/json"],
            body: Data(treeJson.utf8))
    }

    private func serveResolve(
        method: String,
        repository: ScriptedRepository,
        hubFile: ScriptedHubFile,
        rangeHeader: String?,
        connection: NWConnection
    ) {
        self.stateQueue.sync {
            self.resolveRequestCount = self.resolveRequestCount + 1
        }
        if self.stateQueue.sync(execute: { return self.failResolveRequests; }) {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 503 Service Unavailable",
                headers: [:],
                body: Data())
            return
        }
        let effectiveBytes: Data = self.servedBytes(of: hubFile, repositoryId: repository.repositoryId)
        let etagHex: String = hubFile.servesAsLfs
            ? ScriptedHuggingFaceHub.sha256Hex(effectiveBytes)
            : ScriptedHuggingFaceHub.gitBlobSha1Hex(effectiveBytes)
        if method == "HEAD" {
            self.writePlainResponse(
                connection,
                statusLine: "HTTP/1.1 200 OK",
                headers: [
                    "ETag": "\"\(etagHex)\"",
                    "X-Linked-Etag": "\"\(etagHex)\"",
                    "X-Repo-Commit": repository.revision,
                ],
                body: Data(),
                declaredBodyByteCount: effectiveBytes.count)
            return
        }
        var bodyStartOffset: Int = 0
        var statusLine: String = "HTTP/1.1 200 OK"
        var rangeHeaderLine: String? = nil
        if let rangeHeader: String = rangeHeader,
            rangeHeader.hasPrefix("bytes="),
            let rangeOffset: Int = Int(rangeHeader.dropFirst("bytes=".count).split(separator: "-").first ?? ""),
            rangeOffset > 0,
            rangeOffset < effectiveBytes.count
        {
            bodyStartOffset = rangeOffset
            statusLine = "HTTP/1.1 206 Partial Content"
            rangeHeaderLine = "Content-Range: bytes \(rangeOffset)-\(effectiveBytes.count - 1)/\(effectiveBytes.count)"
        }
        let responseBytes: Data = effectiveBytes.subdata(in: bodyStartOffset ..< effectiveBytes.count)
        var responseHeader: String = statusLine + "\r\n"
            + "ETag: \"\(etagHex)\"\r\n"
            + "X-Linked-Etag: \"\(etagHex)\"\r\n"
            + "X-Repo-Commit: \(repository.revision)\r\n"
            + "Content-Type: application/octet-stream\r\n"
            + "Content-Length: \(responseBytes.count)\r\n"
        if let rangeHeaderLine: String = rangeHeaderLine {
            responseHeader += rangeHeaderLine + "\r\n"
        }
        responseHeader += "Connection: keep-alive\r\n\r\n"
        let unsafeConnection: NWConnection = connection
        connection.send(
            content: Data(responseHeader.utf8),
            contentContext: .defaultMessage,
            isComplete: false,
            completion: .contentProcessed { (sendError: NWError?) -> Void in
                self.streamPacedBody(responseBytes, connection: unsafeConnection, offset: 0)
            })
    }

    private func streamPacedBody(_ body: Data, connection: NWConnection, offset: Int) {
        let unsafeConnection: NWConnection = connection
        guard offset < body.count else {
            connection.send(
                content: Data(),
                contentContext: .defaultMessage,
                isComplete: true,
                completion: .contentProcessed { (sendError: NWError?) -> Void in
                })
            return
        }
        let chunkEnd: Int = min(offset + self.chunkByteCount, body.count)
        let chunk: Data = body.subdata(in: offset ..< chunkEnd)
        self.stateQueue.sync {
            self.payloadBytesStreamed = self.payloadBytesStreamed + UInt64(chunk.count)
        }
        let sendChunk: @Sendable () -> Void = {
            connection.send(
                content: chunk,
                contentContext: .defaultMessage,
                isComplete: false,
                completion: .contentProcessed { (sendError: NWError?) -> Void in
                    self.streamPacedBody(body, connection: unsafeConnection, offset: chunkEnd)
                })
        }
        if self.chunkDelayMilliseconds > 0 {
            self.queue.asyncAfter(
                deadline: .now() + Double(self.chunkDelayMilliseconds) / 1000.0,
                execute: sendChunk)
        } else {
            sendChunk()
        }
    }

    // MARK: - Plumbing

    private func repository(_ repositoryId: String) -> ScriptedRepository? {
        return self.repositories.first { (repository: ScriptedRepository) -> Bool in
            return repository.repositoryId == repositoryId
        }
    }

    private func resolveRepositoryAndFilePath(_ path: String) -> (ScriptedRepository, ScriptedHubFile)? {
        for repository: ScriptedRepository in self.repositories {
            let resolvePrefix: String = "/\(repository.repositoryId)/resolve/\(repository.revision)/"
            guard path.hasPrefix(resolvePrefix) else {
                continue
            }
            let relativePath: String = String(path.dropFirst(resolvePrefix.count))
            guard let hubFile: ScriptedHubFile = repository.files.first(where: { (hubFile: ScriptedHubFile) -> Bool in
                return hubFile.relativePath == relativePath
            })
            else {
                continue
            }
            return (repository, hubFile)
        }
        return nil
    }

    private func writePlainResponse(
        _ connection: NWConnection,
        statusLine: String,
        headers: Dictionary<String, String>,
        body: Data,
        declaredBodyByteCount: Int? = nil
    ) {
        var responseText: String = statusLine + "\r\n"
        for (headerName, headerValue) in headers {
            responseText += "\(headerName): \(headerValue)\r\n"
        }
        responseText += "Content-Length: \(declaredBodyByteCount ?? body.count)\r\n"
        responseText += "Connection: keep-alive\r\n\r\n"
        var responseBytes: Data = Data(responseText.utf8)
        responseBytes.append(body)
        connection.send(
            content: responseBytes,
            contentContext: .defaultMessage,
            isComplete: true,
            completion: .contentProcessed { (sendError: NWError?) -> Void in
            })
    }

    private static func sha256Hex(_ bytes: Data) -> String {
        let digest: SHA256.Digest = SHA256.hash(data: bytes)
        return digest.map({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte)
        }).joined()
    }

    private static func gitBlobSha1Hex(_ bytes: Data) -> String {
        var blobContent: Data = Data("blob \(bytes.count)\0".utf8)
        blobContent.append(bytes)
        let digest: Insecure.SHA1.Digest = Insecure.SHA1.hash(data: blobContent)
        return digest.map({ (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte)
        }).joined()
    }
}
