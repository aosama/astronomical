import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for PUT /v1/config/maximum-mlx-memory, migrating
 * apps/supervisor/tests/rest_api/config_reload/maximum_mlx_memory.rs over
 * the shared fake worker: unrelated config changes demand the full reload
 * first, a worker rejection never restores stale bytes over a newer
 * persisted document, and a queued raise that ends rejected preserves the
 * persisted intent both times.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestMaximumMlxMemoryEndpointTests {


    @Test
    func should_require_full_reload_when_other_configuration_changes_are_pending() throws {
        let journey: MaximumMlxMemoryJourney = try MaximumMlxMemoryJourney.launch();
        defer { journey.dispose() }
        MaximumMlxMemoryJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"diagnostics\":{\"log_level\":\"info\"}}");

        let memoryResponse: RestHttpResponse = journey.putMaximumMlxMemory(maximumMlxMemoryGb: 32);

        #expect(memoryResponse.statusCode == 409);
        let persistedConfigBytes: Data = try Data(
            contentsOf: journey.configFileUrl);
        let persistedConfigText: String = String(decoding: persistedConfigBytes, as: UTF8.self);
        #expect(persistedConfigText.contains("log_level"),
            "the operator's diagnostics change must survive the refused memory update");
        #expect(!persistedConfigText.contains("maximum_mlx_memory_gb"),
            "the refused memory update must not persist its candidate document");
        #expect(journey.transitionState.currentReloadableConfig().maximumMlxMemoryBytes == nil);
    }

    @Test
    func should_not_let_a_rejected_update_restore_over_a_newer_memory_setting() throws {
        let journey: MaximumMlxMemoryJourney = try MaximumMlxMemoryJourney.launch();
        defer { journey.dispose() }
        try journey.harness.pokeMemoryRaiseRejection(31_000_000_000);
        let rejectedUpdateOutcome: MemoryUpdateJourneyOutcome = journey.putMaximumMlxMemoryInBackground(
            maximumMlxMemoryGb: 31);

        MaximumMlxMemoryJourney.waitForPersistedMaximum(journey, expectedGigabytes: 31);
        MaximumMlxMemoryJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],\"maximum_mlx_memory_gb\":32}}");
        MaximumMlxMemoryJourney.joinMemoryUpdate(rejectedUpdateOutcome, deadline: Date().addingTimeInterval(10));

        #expect(rejectedUpdateOutcome.observedStatusCode == 400);
        MaximumMlxMemoryJourney.waitForPersistedMaximum(journey, expectedGigabytes: 32);
    }

    @Test
    func should_preserve_persisted_intent_when_queued_application_is_rejected() throws {
        let journey: MaximumMlxMemoryJourney = try MaximumMlxMemoryJourney.launch();
        defer { journey.dispose() }
        let firstGeneration: GenerationJourneyOutcome = journey.harness.startGenerationThread(requestId: 9_001);
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 1);

        let queuedResponse: RestHttpResponse = journey.putMaximumMlxMemory(maximumMlxMemoryGb: 31);
        #expect(queuedResponse.statusCode == 202);
        let conflictingResponse: RestHttpResponse = journey.putMaximumMlxMemory(maximumMlxMemoryGb: 32);
        #expect(conflictingResponse.statusCode == 409);
        try journey.harness.pokeMemoryRaiseRejection(31_000_000_000);
        try journey.harness.pokeCompletion(requestId: 9_001);
        MaximumMlxMemoryJourney.joinGeneration(firstGeneration, deadline: Date().addingTimeInterval(10));
        MaximumMlxMemoryJourney.awaitMemoryRejection(journey);
        MaximumMlxMemoryJourney.waitForPersistedMaximum(journey, expectedGigabytes: 31);

        let secondGeneration: GenerationJourneyOutcome = journey.harness.startGenerationThread(requestId: 9_002);
        try journey.harness.awaitQueueFill(expectedOutstandingCount: 1);
        let secondQueuedResponse: RestHttpResponse = journey.putMaximumMlxMemory(maximumMlxMemoryGb: 31);
        #expect(secondQueuedResponse.statusCode == 202);
        MaximumMlxMemoryJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],\"maximum_mlx_memory_gb\":33}}");
        try journey.harness.pokeMemoryRaiseRejection(31_000_000_000);
        try journey.harness.pokeCompletion(requestId: 9_002);
        MaximumMlxMemoryJourney.joinGeneration(secondGeneration, deadline: Date().addingTimeInterval(10));
        MaximumMlxMemoryJourney.awaitMemoryRejection(journey);
        MaximumMlxMemoryJourney.waitForPersistedMaximum(journey, expectedGigabytes: 33);
    }
}

/// One launched supervisor with its fake worker, a Development-shaped config
/// home, and the serving route table wired with the memory-limit context.
final class MaximumMlxMemoryJourney: @unchecked Sendable {

    let harness: FakeWorkerJourneyHarness;
    let transitionState: ConfigTransitionState;
    let homeDirectoryUrl: URL;
    private let resolver: ResolvedRuntimeConfigResolver;
    private let routeTable: RestRouteTable;

    private init(
        harness: FakeWorkerJourneyHarness,
        transitionState: ConfigTransitionState,
        homeDirectoryUrl: URL,
        resolver: ResolvedRuntimeConfigResolver,
        routeTable: RestRouteTable
    ) {
        self.harness = harness;
        self.transitionState = transitionState;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.resolver = resolver;
        self.routeTable = routeTable;
    }

    static func launch() throws -> MaximumMlxMemoryJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-memory-journey-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        MaximumMlxMemoryJourney.writeConfigFile(homeDirectoryUrl, configuredFieldsJson: "{}");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: FilePath(string: "/bin/bash"));
        let initialResolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: harness.supervisor.ownedHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            memoryContext: RestMaximumMlxMemoryRouteContext(
                workerControl: harness.supervisor,
                runtimeConfigResolver: resolver,
                transitionState: transitionState));
        return MaximumMlxMemoryJourney(
            harness: harness,
            transitionState: transitionState,
            homeDirectoryUrl: homeDirectoryUrl,
            resolver: resolver,
            routeTable: routeTable);
    }

    func dispose() -> Void {
        self.harness.dispose();
        try? FileManager.default.removeItem(at: self.homeDirectoryUrl);
    }

    func putMaximumMlxMemory(maximumMlxMemoryGb: UInt64) -> RestHttpResponse {
        let memoryRequest: RestHttpRequest = MaximumMlxMemoryJourney.memoryRequest(
            maximumMlxMemoryGb: maximumMlxMemoryGb);
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestMaximumMlxMemoryEndpoint.routeMethod,
            path: RestMaximumMlxMemoryEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            return RestHttpResponse.text(statusCode: 599, body: "the memory route is missing");
        }
        return (try? routeHandler(memoryRequest))
            ?? RestHttpResponse.text(statusCode: 599, body: "the memory route handler failed");
    }

    /// Runs one memory update on its own thread, since an idle-worker update
    /// blocks until the fake worker answers with its armed acknowledgement.
    func putMaximumMlxMemoryInBackground(maximumMlxMemoryGb: UInt64) -> MemoryUpdateJourneyOutcome {
        let updateOutcome: MemoryUpdateJourneyOutcome = MemoryUpdateJourneyOutcome(workerThread: Thread());
        let journey: MaximumMlxMemoryJourney = self;
        let updateThread: Thread = Thread {
            let observedResponse: RestHttpResponse = journey.putMaximumMlxMemory(
                maximumMlxMemoryGb: maximumMlxMemoryGb);
            updateOutcome.record(observedResponse);
        };
        updateOutcome.workerThread = updateThread;
        updateThread.name = "journey-memory-update";
        updateThread.start();
        return updateOutcome;
    }

    var configFileUrl: URL {
        return self.homeDirectoryUrl.appendingPathComponent(".astronomical-dev/config.json");
    }

    // MARK: Fixture helpers

    /// Merges the configured fields over the minimal v1 document and writes
    /// the instance config file, mirroring the Rust support's
    /// write_config_file helper.
    static func writeConfigFile(_ homeDirectoryUrl: URL, configuredFieldsJson: String) -> Void {
        let configuredFields: [String: Any];
        if let parsedFields: [String: Any] = (try? JSONSerialization.jsonObject(
            with: Data(configuredFieldsJson.utf8), options: [])) as? [String: Any] {
            configuredFields = parsedFields;
        } else {
            configuredFields = [:];
        }
        var configDocument: [String: Any] = [
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": ["model_directories": Array<String>()],
        ];
        for (fieldName, fieldValue) in configuredFields {
            configDocument[fieldName] = fieldValue;
        }
        let configDocumentBytes: Data;
        if JSONSerialization.isValidJSONObject(configDocument) {
            configDocumentBytes = (try? JSONSerialization.data(
                withJSONObject: configDocument,
                options: [.prettyPrinted, .sortedKeys])) ?? Data();
        } else {
            configDocumentBytes = Data();
        }
        let configFileUrl: URL = homeDirectoryUrl.appendingPathComponent(".astronomical-dev/config.json");
        try? FileManager.default.createDirectory(
            at: configFileUrl.deletingLastPathComponent(),
            withIntermediateDirectories: true);
        try? configDocumentBytes.write(to: configFileUrl);
    }

    static func memoryRequest(maximumMlxMemoryGb: UInt64) -> RestHttpRequest {
        let memoryBody: Data = Data("{\"maximum_mlx_memory_gb\":\(maximumMlxMemoryGb)}".utf8);
        return RestHttpRequest(
            method: RestMaximumMlxMemoryEndpoint.routeMethod,
            path: RestMaximumMlxMemoryEndpoint.routePath,
            requestTarget: RestMaximumMlxMemoryEndpoint.routePath,
            headersByLowercasedName: ["content-type": "application/json"],
            bodyBytes: memoryBody);
    }

    /// Blocks until the persisted document carries the expected memory
    /// setting, bounding the wait so the journey carries its own deadline.
    static func waitForPersistedMaximum(_ journey: MaximumMlxMemoryJourney, expectedGigabytes: UInt64) -> Void {
        let persistenceDeadline: Date = Date().addingTimeInterval(2);
        while true {
            if let persistedGigabytes: UInt64 = MaximumMlxMemoryJourney.readPersistedMaximum(journey),
                persistedGigabytes == expectedGigabytes {
                return;
            }
            #expect(Date() < persistenceDeadline,
                "maximum_mlx_memory_gb did not become \(expectedGigabytes)");
            if Date() >= persistenceDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    private static func readPersistedMaximum(_ journey: MaximumMlxMemoryJourney) -> UInt64? {
        guard let persistedConfigBytes: Data = try? Data(contentsOf: journey.configFileUrl),
            let persistedConfigDocument: Any = try? JSONSerialization.jsonObject(
                with: persistedConfigBytes, options: []),
            let persistedConfigObject: [String: Any] = persistedConfigDocument as? [String: Any],
            let runtimeDocument: [String: Any] = persistedConfigObject["runtime"] as? [String: Any] else {
            return nil;
        }
        return runtimeDocument["maximum_mlx_memory_gb"] as? UInt64;
    }

    /// Blocks until the deferred memory raise ends in a published worker
    /// rejection, bounding the wait exactly as wait_for_memory_rejection.
    static func awaitMemoryRejection(_ journey: MaximumMlxMemoryJourney) -> Void {
        let rejectionDeadline: Date = Date().addingTimeInterval(2);
        while true {
            let workerHealthSnapshot: WorkerHealthSnapshot = journey.harness.supervisor.workerHealthSnapshot();
            if workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil
                && workerHealthSnapshot.mlxMemoryLimitError != nil {
                Thread.sleep(forTimeInterval: 0.05);
                return;
            }
            #expect(Date() < rejectionDeadline, "the queued memory rejection was never published");
            if Date() >= rejectionDeadline {
                return;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    static func joinGeneration(_ outcome: GenerationJourneyOutcome, deadline: Date) -> Void {
        while outcome.workerThread.isFinished == false && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        #expect(outcome.workerThread.isFinished, "the generation thread outlived its deadline");
    }

    static func joinMemoryUpdate(_ outcome: MemoryUpdateJourneyOutcome, deadline: Date) -> Void {
        while outcome.workerThread.isFinished == false && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        #expect(outcome.workerThread.isFinished, "the memory-update thread outlived its deadline");
    }
}

/// One memory update executed on its own thread, capturing the response the
/// journey thread inspects after a bounded join.
final class MemoryUpdateJourneyOutcome: @unchecked Sendable {

    var workerThread: Thread;
    private let outcomeLock: NSLock;
    private var observedStatusCodeValue: Int?;

    init(workerThread: Thread) {
        self.workerThread = workerThread;
        self.outcomeLock = NSLock();
        self.observedStatusCodeValue = nil;
    }

    var observedStatusCode: Int? {
        self.outcomeLock.lock();
        defer { self.outcomeLock.unlock(); }
        return self.observedStatusCodeValue;
    }

    func record(_ observedResponse: RestHttpResponse) -> Void {
        self.outcomeLock.lock();
        self.observedStatusCodeValue = observedResponse.statusCode;
        self.outcomeLock.unlock();
    }
}
