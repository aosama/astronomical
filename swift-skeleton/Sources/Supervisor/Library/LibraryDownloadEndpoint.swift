import Foundation

import IpcProtocol

/**
 * The Library download REST controls: start and read one download, pause,
 * resume, and cancel it. The response projection and error envelope mirror
 * the Rust download endpoint contract exactly — path-free error codes and
 * idle semantics included.
 */
public enum LibraryDownloadEndpoint {

    public static let downloadRoutePath: String = "/v1/library/download"
    public static let pauseRoutePath: String = "/v1/library/download/pause"
    public static let resumeRoutePath: String = "/v1/library/download/resume"
    public static let cancelRoutePath: String = "/v1/library/download/cancel"

    private static let MAXIMUM_DOWNLOAD_CONTROL_BODY_BYTES: Int = 4_096

    public static func currentDownloadResponse(context: RestLibraryDownloadRouteContext) throws -> RestHttpResponse {
        return try RestAsyncBridge.response {
            guard let record: LibraryDownloadJobRecord = await context.coordinator.currentJob() else {
                return LibraryDownloadEndpoint.idleResponse()
            }
            return LibraryDownloadEndpoint.response(
                statusCode: 200,
                state: record.state.rawValue,
                huggingfaceId: record.huggingfaceId,
                revision: record.revision,
                bytesCompleted: record.bytesCompleted,
                bytesTotal: record.bytesTotal,
                currentFileRelativePath: nil,
                destinationDirectory: context.coordinator.destinationDirectory(
                    huggingfaceId: record.huggingfaceId).string,
                errorCode: record.errorCode)
        }
    }

    public static func startDownloadResponse(
        request: RestHttpRequest,
        context: RestLibraryDownloadRouteContext
    ) throws -> RestHttpResponse {
        guard let huggingfaceId: String = try LibraryDownloadEndpoint.decodeStartRequest(request) else {
            return LibraryDownloadEndpoint.errorResponse(
                statusCode: 400,
                errorCode: "download_failed",
                message: "The download request body is not a valid huggingface_id document.")
        }
        return try RestAsyncBridge.response {
            do {
                try await context.coordinator.start(huggingfaceId: huggingfaceId)
                return LibraryDownloadEndpoint.response(
                    statusCode: 202,
                    state: "checking_disk",
                    huggingfaceId: huggingfaceId,
                    revision: nil,
                    bytesCompleted: 0,
                    bytesTotal: 0,
                    currentFileRelativePath: nil,
                    destinationDirectory: nil,
                    errorCode: nil)
            } catch let coordinatorError as LibraryDownloadCoordinator.CoordinatorError {
                return LibraryDownloadEndpoint.coordinatorErrorResponse(coordinatorError)
            }
        }
    }

    public static func pauseResponse(context: RestLibraryDownloadRouteContext) throws -> RestHttpResponse {
        return try RestAsyncBridge.response {
            do {
                let record: LibraryDownloadJobRecord = try await context.coordinator.pause()
                return LibraryDownloadEndpoint.response(
                    statusCode: 200,
                    state: record.state.rawValue,
                    huggingfaceId: record.huggingfaceId,
                    revision: record.revision,
                    bytesCompleted: record.bytesCompleted,
                    bytesTotal: record.bytesTotal,
                    currentFileRelativePath: nil,
                    destinationDirectory: context.coordinator.destinationDirectory(
                        huggingfaceId: record.huggingfaceId).string,
                    errorCode: record.errorCode)
            } catch let coordinatorError as LibraryDownloadCoordinator.CoordinatorError {
                return LibraryDownloadEndpoint.coordinatorErrorResponse(coordinatorError)
            }
        }
    }

    public static func resumeResponse(context: RestLibraryDownloadRouteContext) throws -> RestHttpResponse {
        return try RestAsyncBridge.response {
            do {
                try await context.coordinator.resume()
                return LibraryDownloadEndpoint.response(
                    statusCode: 202,
                    state: "resuming",
                    huggingfaceId: nil,
                    revision: nil,
                    bytesCompleted: 0,
                    bytesTotal: 0,
                    currentFileRelativePath: nil,
                    destinationDirectory: nil,
                    errorCode: nil)
            } catch let coordinatorError as LibraryDownloadCoordinator.CoordinatorError {
                return LibraryDownloadEndpoint.coordinatorErrorResponse(coordinatorError)
            }
        }
    }

    public static func cancelResponse(context: RestLibraryDownloadRouteContext) throws -> RestHttpResponse {
        return try RestAsyncBridge.response {
            await context.coordinator.cancel()
            return LibraryDownloadEndpoint.idleResponse()
        }
    }

    // MARK: - Projection

    private static func idleResponse() -> RestHttpResponse {
        return LibraryDownloadEndpoint.response(
            statusCode: 200,
            state: "idle",
            huggingfaceId: nil,
            revision: nil,
            bytesCompleted: 0,
            bytesTotal: 0,
            currentFileRelativePath: nil,
            destinationDirectory: nil,
            errorCode: nil)
    }

    private static func response(
        statusCode: Int,
        state: String,
        huggingfaceId: String?,
        revision: String?,
        bytesCompleted: UInt64,
        bytesTotal: UInt64,
        currentFileRelativePath: String?,
        destinationDirectory: String?,
        errorCode: String?
    ) -> RestHttpResponse {
        var responseObject: JsonWireObject = JsonWireObject(entries: [])
        responseObject.appendEntry(key: "state", value: .string(state))
        responseObject.appendEntry(
            key: "huggingface_id",
            value: huggingfaceId.map({ (identity: String) -> JsonWireValue in return .string(identity) }) ?? .null)
        responseObject.appendEntry(
            key: "revision",
            value: revision.map({ (pinnedRevision: String) -> JsonWireValue in return .string(pinnedRevision) }) ?? .null)
        responseObject.appendEntry(key: "bytes_completed", value: .unsignedInteger(bytesCompleted))
        responseObject.appendEntry(key: "bytes_total", value: .unsignedInteger(bytesTotal))
        responseObject.appendEntry(
            key: "current_file_relative_path",
            value: currentFileRelativePath.map({ (relativePath: String) -> JsonWireValue in return .string(relativePath) }) ?? .null)
        if let destinationDirectory: String = destinationDirectory {
            responseObject.appendEntry(key: "destination_directory", value: .string(destinationDirectory))
        }
        responseObject.appendEntry(
            key: "error_code",
            value: errorCode.map({ (codeName: String) -> JsonWireValue in return .string(codeName) }) ?? .null)
        return (try? RestHttpResponse.json(statusCode: statusCode, wireValue: .object(responseObject)))
            ?? RestHttpResponse.text(statusCode: 500, body: "download projection failure")
    }

    private static func coordinatorErrorResponse(
        _ coordinatorError: LibraryDownloadCoordinator.CoordinatorError
    ) -> RestHttpResponse {
        switch coordinatorError {
        case .libraryBusy:
            return LibraryDownloadEndpoint.errorResponse(
                statusCode: 409,
                errorCode: "library_busy",
                message: "Another model download is already active.")
        case .catalogEntryNotFound:
            return LibraryDownloadEndpoint.errorResponse(
                statusCode: 404,
                errorCode: "catalog_entry_not_found",
                message: "That model is not available in this release catalog.")
        case .jobNotFound:
            return LibraryDownloadEndpoint.errorResponse(
                statusCode: 404,
                errorCode: "download_failed",
                message: "There is no model download to control.")
        }
    }

    private static func errorResponse(
        statusCode: Int,
        errorCode: String,
        message: String
    ) -> RestHttpResponse {
        var errorObject: JsonWireObject = JsonWireObject(entries: [])
        errorObject.appendEntry(key: "code", value: .string(errorCode))
        errorObject.appendEntry(key: "message", value: .string(message))
        var responseObject: JsonWireObject = JsonWireObject(entries: [])
        responseObject.appendEntry(key: "error", value: .object(errorObject))
        return (try? RestHttpResponse.json(statusCode: statusCode, wireValue: .object(responseObject)))
            ?? RestHttpResponse.text(statusCode: 500, body: "download error projection failure")
    }

    private static func decodeStartRequest(_ request: RestHttpRequest) throws -> String? {
        guard request.bodyBytes.count <= LibraryDownloadEndpoint.MAXIMUM_DOWNLOAD_CONTROL_BODY_BYTES else {
            return nil
        }
        guard let bodyDocument: Any = try? JSONSerialization.jsonObject(with: request.bodyBytes, options: []) else {
            return nil
        }
        guard let bodyObject: [String: Any] = bodyDocument as? [String: Any] else {
            return nil
        }
        guard bodyObject.count == 1,
            let huggingfaceId: String = bodyObject["huggingface_id"] as? String
        else {
            return nil
        }
        return huggingfaceId
    }
}

/// The coordinator handle the download routes answer from.
public struct RestLibraryDownloadRouteContext: Sendable {
    public let coordinator: LibraryDownloadCoordinator

    public init(coordinator: LibraryDownloadCoordinator) {
        self.coordinator = coordinator
    }
}

/**
 * Bridges an async coordinator operation into the synchronous per-connection
 * handler thread. Each REST connection owns its own thread, so blocking here
 * can never stall an executor, and the semaphore bounds the wait to the
 * operation itself.
 */
enum RestAsyncBridge {

    private final class OutcomeBox: @unchecked Sendable {
        private let stateLock: NSLock = NSLock()
        private var outcome: Result<RestHttpResponse, Error>?

        func record(_ outcome: Result<RestHttpResponse, Error>) {
            self.stateLock.lock()
            self.outcome = outcome
            self.stateLock.unlock()
        }

        func take() -> Result<RestHttpResponse, Error>? {
            self.stateLock.lock()
            let currentOutcome: Result<RestHttpResponse, Error>? = self.outcome
            self.stateLock.unlock()
            return currentOutcome
        }
    }

    static func response(
        _ operation: @escaping @Sendable () async throws -> RestHttpResponse
    ) throws -> RestHttpResponse {
        let outcomeBox: OutcomeBox = OutcomeBox()
        let completionSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0)
        Task<Void, Never> {
            do {
                outcomeBox.record(.success(try await operation()))
            } catch {
                outcomeBox.record(.failure(error))
            }
            completionSemaphore.signal()
        }
        completionSemaphore.wait()
        guard let outcome: Result<RestHttpResponse, Error> = outcomeBox.take() else {
            throw RestEndpointFailure.internalError(message: "the download control bridge lost its outcome")
        }
        return try outcome.get()
    }
}
