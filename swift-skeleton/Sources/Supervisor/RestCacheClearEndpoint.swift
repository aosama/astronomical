import Foundation;

import IpcProtocol;
import RestContract;

/// The context the daemon hands the cache-clear route: the live supervisor's
/// clear control, or nil when no worker control exists and the route must
/// advertise itself as absent.
public struct RestCacheClearRouteContext: @unchecked Sendable {

    let cacheClearExecutor: any PromptCacheClearControlling;

    public init(cacheClearExecutor: any PromptCacheClearControlling) {
        self.cacheClearExecutor = cacheClearExecutor;
    }
}

/// Deletes global or model-scoped SSD prompt-cache content through the
/// worker, migrating apps/supervisor/src/cache_clear_endpoint.rs:
/// DELETE /v1/cache?model={id} answers 200 cleared, 202 queued, 404 without
/// live worker control, 400 for a model id that could escape the cache root,
/// and 503 when the worker control fails.
public enum RestCacheClearEndpoint {

    public static let routeMethod: String = "DELETE";
    public static let routePath: String = "/v1/cache";

    public static func handle(
        _ request: RestHttpRequest,
        cacheClearContext: RestCacheClearRouteContext?
    ) throws -> RestHttpResponse {
        guard let cacheClearContext = cacheClearContext else {
            return RestHttpResponse.text(statusCode: 404, body: "live worker control is unavailable");
        }
        let requestedModelId: String? = RestCacheClearEndpoint.queryModelId(request);
        if let requestedModelId = requestedModelId,
            !RestCacheClearEndpoint.isSafeModelId(requestedModelId) {
            return RestHttpResponse.text(
                statusCode: 400,
                body: "model must be a safe relative model ID");
        }
        do {
            switch (try cacheClearContext.cacheClearExecutor.clearPromptCache(modelId: requestedModelId)) {
            case let .applied(clearedModelId, blocksRemoved, bytesFreed):
                return try RestCacheClearEndpoint.clearResponse(
                    statusCode: 200,
                    status: "cleared",
                    modelId: clearedModelId,
                    blocksRemoved: blocksRemoved,
                    bytesFreed: bytesFreed);
            case .queued:
                return try RestCacheClearEndpoint.clearResponse(
                    statusCode: 202,
                    status: "queued",
                    modelId: requestedModelId,
                    blocksRemoved: 0,
                    bytesFreed: 0);
            }
        } catch {
            return RestHttpResponse.text(statusCode: 503, body: "worker cache clear failed");
        }
    }

    /// The decoded `model` query parameter, distinguishing an absent
    /// parameter from the explicit empty value the safety rule rejects.
    static func queryModelId(_ request: RestHttpRequest) -> String? {
        guard let queryStart: Range<String.Index> = request.requestTarget.range(of: "?") else {
            return nil;
        }
        let query: String = String(request.requestTarget[queryStart.upperBound...]);
        for queryPair: Substring in query.split(separator: "&") {
            let queryPairParts: [Substring] = queryPair.split(
                separator: "=",
                maxSplits: 1,
                omittingEmptySubsequences: false);
            guard queryPairParts.first == "model" else {
                continue;
            }
            let rawModelValue: String = queryPairParts.count > 1 ? String(queryPairParts[1]) : "";
            return rawModelValue.removingPercentEncoding ?? rawModelValue;
        }
        return nil;
    }

    /// Rejects model ids that are not a plain relative path: empty, control
    /// characters, separators, or any dot component that could climb outside
    /// the cache root, mirroring cache_clear_endpoint.rs's is_safe_model_id.
    static func isSafeModelId(_ modelId: String) -> Bool {
        if modelId.isEmpty || modelId.contains("\0") || modelId.contains("\\") {
            return false;
        }
        if modelId.hasPrefix("/") {
            return false;
        }
        for modelIdComponent: Substring in modelId.split(separator: "/") {
            if modelIdComponent == "." || modelIdComponent == ".." {
                return false;
            }
        }
        return true;
    }

    private static func clearResponse(
        statusCode: Int,
        status: String,
        modelId: String?,
        blocksRemoved: UInt64,
        bytesFreed: UInt64
    ) throws -> RestHttpResponse {
        var clearDocument: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        clearDocument.appendEntry(key: "status", value: .string(status));
        clearDocument.appendEntry(
            key: "model_id",
            value: modelId.map({ (clearedModelId: String) -> JsonWireValue in
                return .string(clearedModelId);
            }) ?? .null);
        clearDocument.appendEntry(key: "blocks_removed", value: .unsignedInteger(blocksRemoved));
        clearDocument.appendEntry(key: "bytes_freed", value: .unsignedInteger(bytesFreed));
        return try RestHttpResponse.json(statusCode: statusCode, wireValue: .object(clearDocument));
    }
}
