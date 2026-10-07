import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the DELETE /v1/cache surface, migrating
 * apps/supervisor/tests/rest_api/cache_clear.rs over the shared fake worker:
 * idle clears apply synchronously, busy clears queue as the newest pending
 * deletion until the whole admission queue drains, unsafe model ids are
 * refused, and every failure maps to its stable status code.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestCacheClearEndpointTests {

    @Test
    func should_clear_the_entire_cache_while_the_worker_is_idle() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        try journey.harness.pokeCacheClearAck(scopeModelId: nil, blocksRemoved: 3, bytesFreed: 4096);

        let clearResponse: RestHttpResponse = try journey.deleteCache(modelQuery: nil);

        #expect(clearResponse.statusCode == 200);
        let clearDocument: [String: Any] = try CacheClearJourney.decodeObject(clearResponse);
        #expect(clearDocument["status"] as? String == "cleared");
        #expect(clearDocument["model_id"] is NSNull);
        #expect(clearDocument["blocks_removed"] as? UInt64 == 3);
        #expect(clearDocument["bytes_freed"] as? UInt64 == 4096);
        #expect(journey.harness.supervisor.workerHealthSnapshot().pendingPromptCacheClear == nil);
    }

    @Test
    func should_clear_one_model_cache_by_its_model_id() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        try journey.harness.pokeCacheClearAck(
            scopeModelId: CacheClearJourney.requestedModelId,
            blocksRemoved: 2,
            bytesFreed: 2048);

        let clearResponse: RestHttpResponse = try journey.deleteCache(
            modelQuery: CacheClearJourney.requestedModelId);

        #expect(clearResponse.statusCode == 200);
        let clearDocument: [String: Any] = try CacheClearJourney.decodeObject(clearResponse);
        #expect(clearDocument["status"] as? String == "cleared");
        #expect(clearDocument["model_id"] as? String == CacheClearJourney.requestedModelId);
    }

    @Test
    func should_queue_only_the_newest_cache_clear_until_generation_finishes() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        let activeOutcome: GenerationJourneyOutcome = journey.harness.startGenerationThread(requestId: 41);
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 1);

        let globalClearResponse: RestHttpResponse = try journey.deleteCache(modelQuery: nil);
        #expect(globalClearResponse.statusCode == 202);
        let globalClearDocument: [String: Any] = try CacheClearJourney.decodeObject(globalClearResponse);
        #expect(globalClearDocument["status"] as? String == "queued");
        #expect(globalClearDocument["model_id"] is NSNull);

        let scopedClearResponse: RestHttpResponse = try journey.deleteCache(
            modelQuery: CacheClearJourney.requestedModelId);
        #expect(scopedClearResponse.statusCode == 202);

        let queuedStatsDocument: [String: Any] = try journey.getCacheStats();
        let pendingCacheClear: [String: Any] = try CacheClearJourney.requirePendingCacheClear(
            queuedStatsDocument);
        #expect(pendingCacheClear["model_id"] as? String == CacheClearJourney.requestedModelId);

        // The newest clear applies when the generation finalizes and no
        // waiter remains.
        try journey.harness.pokeCacheClearAck(
            scopeModelId: CacheClearJourney.requestedModelId,
            blocksRemoved: 5,
            bytesFreed: 8192);
        try journey.harness.pokeCompletion(requestId: 41);
        CacheClearJourney.join(activeOutcome, deadline: Date().addingTimeInterval(10));
        CacheClearJourney.awaitPendingClearFinished(journey: journey);
    }

    @Test
    func should_wait_for_a_queued_generation_before_applying_the_cache_clear() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        let firstOutcome: GenerationJourneyOutcome = journey.harness.startGenerationThread(requestId: 51);
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 1);
        let queuedOutcome: GenerationJourneyOutcome = journey.harness.startGenerationThread(requestId: 52);
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 2);

        let clearResponse: RestHttpResponse = try journey.deleteCache(modelQuery: nil);
        #expect(clearResponse.statusCode == 202);

        try journey.harness.pokeCacheClearAck(scopeModelId: nil, blocksRemoved: 3, bytesFreed: 4096);
        try journey.harness.pokeCompletion(requestId: 51);
        CacheClearJourney.join(firstOutcome, deadline: Date().addingTimeInterval(10));

        // The clear must remain pending while the queued generation runs.
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 1);
        let duringQueuedStats: [String: Any] = try journey.getCacheStats();
        #expect(!CacheClearJourney.pendingCacheClearIsNull(duringQueuedStats),
            "cache clear must remain pending while the queued generation runs");

        try journey.harness.pokeCompletion(requestId: 52);
        CacheClearJourney.join(queuedOutcome, deadline: Date().addingTimeInterval(10));
        CacheClearJourney.awaitPendingClearFinished(journey: journey);
    }

    @Test
    func should_reject_a_model_id_that_can_escape_the_cache_root() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }

        let escapingResponse: RestHttpResponse = try journey.deleteCache(modelQuery: "../outside");
        #expect(escapingResponse.statusCode == 400);
        let emptyModelResponse: RestHttpResponse = try journey.deleteCache(modelQuery: "");
        #expect(emptyModelResponse.statusCode == 400);
    }

    @Test
    func should_not_offer_cache_clear_without_live_worker_control() throws {
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: WorkerHealthState(),
            instancePaths: RestChatJourneySupport.journeyInstancePaths(),
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity());

        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "DELETE",
            path: RestCacheClearEndpoint.routePath);
        guard case .notFound = routeOutcome else {
            Issue.record("the route must be absent without worker control, got \(routeOutcome)");
            return;
        }
    }

    @Test
    func should_return_service_unavailable_when_the_worker_has_stopped() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        _ = try journey.harness.supervisor.shutdown();

        let clearResponse: RestHttpResponse = try journey.deleteCache(modelQuery: nil);

        #expect(clearResponse.statusCode == 503);
    }

    @Test
    func should_reject_a_worker_acknowledgement_for_a_different_model_scope() throws {
        let journey: CacheClearJourney = try CacheClearJourney.launch();
        defer { journey.dispose() }
        try journey.harness.pokeCacheClearAck(
            scopeModelId: "astronomical/mismatched-clear-model",
            blocksRemoved: 1,
            bytesFreed: 1024);

        let clearResponse: RestHttpResponse = try journey.deleteCache(
            modelQuery: CacheClearJourney.requestedModelId);

        #expect(clearResponse.statusCode == 503);
    }
}

/// One launched supervisor with its fake worker plus the serving route table
/// over the live health state.
final class CacheClearJourney {

    static let requestedModelId: String = "astronomical/requested-model";

    let harness: FakeWorkerJourneyHarness;
    private let routeTable: RestRouteTable;

    private init(harness: FakeWorkerJourneyHarness, routeTable: RestRouteTable) {
        self.harness = harness;
        self.routeTable = routeTable;
    }

    static func launch() throws -> CacheClearJourney {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: harness.supervisor.ownedHealthState(),
            instancePaths: RestChatJourneySupport.journeyInstancePaths(),
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            cacheClearContext: RestCacheClearRouteContext(cacheClearExecutor: harness.supervisor));
        return CacheClearJourney(harness: harness, routeTable: routeTable);
    }

    func dispose() -> Void {
        self.harness.dispose();
    }

    func deleteCache(modelQuery: String?) throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: "DELETE",
            path: RestCacheClearEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw CacheClearJourneyFailure.cacheRouteMissing(routeOutcome);
        }
        return try routeHandler(CacheClearJourney.clearRequest(modelQuery: modelQuery));
    }

    func getCacheStats() throws -> [String: Any] {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: "GET",
            path: "/v1/cache/stats");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw CacheClearJourneyFailure.statsRouteMissing(routeOutcome);
        }
        let statsRequest: RestHttpRequest = RestHttpRequest(
            method: "GET",
            path: "/v1/cache/stats",
            requestTarget: "/v1/cache/stats",
            headersByLowercasedName: [:],
            bodyBytes: Data());
        return try CacheClearJourney.decodeObject(try routeHandler(statsRequest));
    }

    private static func clearRequest(modelQuery: String?) -> RestHttpRequest {
        let requestTarget: String;
        if let modelQuery = modelQuery {
            let encodedModelQuery: String = modelQuery.addingPercentEncoding(
                withAllowedCharacters: .urlQueryAllowed) ?? modelQuery;
            requestTarget = "\(RestCacheClearEndpoint.routePath)?model=\(encodedModelQuery)";
        } else {
            requestTarget = RestCacheClearEndpoint.routePath;
        }
        return RestHttpRequest(
            method: "DELETE",
            path: RestCacheClearEndpoint.routePath,
            requestTarget: requestTarget,
            headersByLowercasedName: [:],
            bodyBytes: Data());
    }

    static func decodeObject(_ response: RestHttpResponse) throws -> [String: Any] {
        let responseDocument: Any = try JSONSerialization.jsonObject(
            with: response.bodyBytes,
            options: []);
        guard let responseDocumentObject = responseDocument as? [String: Any] else {
            throw CacheClearJourneyFailure.responseNotAnObject(response.statusCode);
        }
        return responseDocumentObject;
    }

    static func requirePendingCacheClear(_ statsDocument: [String: Any]) throws -> [String: Any] {
        guard let pendingCacheClear = statsDocument["pending_cache_clear"] as? [String: Any] else {
            throw CacheClearJourneyFailure.pendingCacheClearMissing;
        }
        return pendingCacheClear;
    }

    static func pendingCacheClearIsNull(_ statsDocument: [String: Any]) -> Bool {
        return statsDocument["pending_cache_clear"] is NSNull;
    }

    /// Blocks until no pending clear remains, bounding the wait so the
    /// journey carries its own deadline.
    static func awaitPendingClearFinished(journey: CacheClearJourney) -> Void {
        let clearDeadline: Date = Date().addingTimeInterval(5);
        while true {
            if journey.harness.supervisor.workerHealthSnapshot().pendingPromptCacheClear == nil {
                return;
            }
            #expect(Date() < clearDeadline, "the queued cache clear was never applied");
            if Date() >= clearDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    static func join(_ outcome: GenerationJourneyOutcome, deadline: Date) -> Void {
        while outcome.workerThread.isFinished == false && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        #expect(outcome.workerThread.isFinished, "the generation thread outlived its deadline");
    }
}

/// Typed failures of the cache-clear journey plumbing.
enum CacheClearJourneyFailure: Error {

    case cacheRouteMissing(RestRouteOutcome);
    case statsRouteMissing(RestRouteOutcome);
    case responseNotAnObject(Int);
    case pendingCacheClearMissing;
}
