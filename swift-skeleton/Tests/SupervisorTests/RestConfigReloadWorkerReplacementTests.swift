import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Acceptance journeys for config reload's transactional worker replacement,
 * migrating apps/supervisor/tests/rest_api/config_reload/
 * transactional_replacement.rs and generation_admission.rs: a candidate that
 * acknowledges a foreign configuration generation is rejected without
 * changing the effective serving state, and configuration control stays
 * responsive while an admitted generation blocks inside its executor.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestConfigReloadWorkerReplacementTests {

    /// The replacement candidate that acknowledges a foreign generation must
    /// be rejected, leaving the previous worker serving with its original
    /// generation and the live config untouched.
    @Test
    func should_reject_mismatched_candidate_without_changing_effective_serving_state() throws {
        let journey: WorkerReplacementJourney = try WorkerReplacementJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"prompt_cache\":{\"maximum_size_gb\":49}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 500);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        let candidateGeneration: String = try WorkerReplacementJourney.requireGeneration(
            reloadDocument, field: "candidate_generation");
        let effectiveGeneration: String = try WorkerReplacementJourney.requireGeneration(
            reloadDocument, field: "effective_generation");
        #expect(candidateGeneration != effectiveGeneration);
        #expect(effectiveGeneration == journey.initialGeneration);
        let retainedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();
        #expect(retainedHealth.status == .ready);
        #expect(
            retainedHealth.workerRuntimeFeatureConfiguration?.configurationGeneration
                == journey.initialGeneration);
        #expect(journey.transitionState.currentReloadableConfig() == journey.initialResolvedConfig);
    }

    /// Configuration control must answer while an admitted generation blocks
    /// inside its executor, and the released generation must still complete.
    @Test
    func should_release_configuration_transition_after_request_queue_admission() throws {
        let journey: AdmissionReloadJourney = try AdmissionReloadJourney.launch();
        defer { journey.dispose() }

        let chatOutcome: AdmissionReloadJourney.ThreadedOutcome = journey.postChatOnThread();
        try journey.awaitAdmissionStarted(within: 2);
        let reloadOutcome: AdmissionReloadJourney.ThreadedOutcome = journey.postConfigReloadOnThread();
        try journey.awaitCompletion(of: reloadOutcome, within: 2);
        journey.releaseAdmission();
        try journey.awaitCompletion(of: chatOutcome, within: 5);

        #expect(chatOutcome.httpResponse?.statusCode == 200);
    }
}

/// One worker-replacement journey: a resolver-driven config home, a real
/// idle fixture worker honoring its startup generation, and the reload
/// route wired with that live supervisor as worker control.
final class WorkerReplacementJourney {

    let supervisor: WorkerSupervisor;
    let transitionState: ConfigTransitionState;
    let initialResolvedConfig: ResolvedRuntimeConfig;
    let initialGeneration: String;
    let homeDirectoryUrl: URL;
    private let routeTable: RestRouteTable;
    private let fixtureDirectoryUrl: URL;

    private init(
        supervisor: WorkerSupervisor,
        transitionState: ConfigTransitionState,
        initialResolvedConfig: ResolvedRuntimeConfig,
        initialGeneration: String,
        homeDirectoryUrl: URL,
        routeTable: RestRouteTable,
        fixtureDirectoryUrl: URL
    ) {
        self.supervisor = supervisor;
        self.transitionState = transitionState;
        self.initialResolvedConfig = initialResolvedConfig;
        self.initialGeneration = initialGeneration;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.routeTable = routeTable;
        self.fixtureDirectoryUrl = fixtureDirectoryUrl;
    }

    static func launch() throws -> WorkerReplacementJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-worker-replacement-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let fixtureDirectoryUrl: URL = homeDirectoryUrl.appendingPathComponent("fixtures", isDirectory: true);
        try FileManager.default.createDirectory(at: fixtureDirectoryUrl, withIntermediateDirectories: true);
        ConfigReloadJourney.writeConfigFile(homeDirectoryUrl, configuredFieldsJson: "{}");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        // The reload resolver points at the mismatched replacement fixture,
        // so the reload candidate carries the candidate executable the
        // journey under tests (mirroring the Rust resolver's
        // for_development_home_directory parameter).
        let replacementWorkerPath: String = try WorkerReplacementJourney.writeFixtureScript(
            fixtureDirectoryUrl: fixtureDirectoryUrl,
            fixtureName: "astronomical-supervisor-replacement-ready-worker",
            acknowledgedGeneration: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: FilePath(string: replacementWorkerPath));
        let initialResolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        let initialGeneration: String = initialResolvedConfig.configurationGeneration;
        let idleWorkerPath: String = try WorkerReplacementJourney.writeFixtureScript(
            fixtureDirectoryUrl: fixtureDirectoryUrl,
            fixtureName: "astronomical-supervisor-idle-worker",
            acknowledgedGeneration: initialGeneration);
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: idleWorkerPath,
            workerArguments: Array<String>(),
            workerStartupConfiguration: initialResolvedConfig.workerStartupConfiguration(),
            modelPolicyCatalog: initialResolvedConfig.modelPolicyCatalog,
            modelLoadTimeout: 2);
        try WorkerReplacementJourney.waitForEffectiveGeneration(
            supervisor: supervisor,
            expectedGeneration: initialGeneration);
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: supervisor.ownedHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: resolver,
                workerControl: supervisor,
                workerHealthState: supervisor.ownedHealthState(),
                generationActivityIdleProvider: { return true }));
        return WorkerReplacementJourney(
            supervisor: supervisor,
            transitionState: transitionState,
            initialResolvedConfig: initialResolvedConfig,
            initialGeneration: initialGeneration,
            homeDirectoryUrl: homeDirectoryUrl,
            routeTable: routeTable,
            fixtureDirectoryUrl: fixtureDirectoryUrl);
    }

    func dispose() -> Void {
        _ = try? self.supervisor.shutdown();
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func postConfigReload() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw WorkerReplacementJourneyFailure.reloadRouteMissing;
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath));
    }

    static func emptyRequest(method: String, path: String) -> RestHttpRequest {
        return RestHttpRequest(
            method: method,
            path: path,
            requestTarget: path,
            headersByLowercasedName: [:],
            bodyBytes: Data());
    }

    static func requireGeneration(_ document: [String: Any], field: String) throws -> String {
        guard let generationValue: String = document[field] as? String else {
            throw WorkerReplacementJourneyFailure.generationMissing(field);
        }
        return generationValue;
    }

    /// Blocks until the live worker reports the expected effective
    /// generation, mirroring the Rust wait_for_effective_generation helper.
    static func waitForEffectiveGeneration(
        supervisor: WorkerSupervisor,
        expectedGeneration: String
    ) throws -> Void {
        let readinessDeadline: Date = Date().addingTimeInterval(2);
        while true {
            let healthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
            if healthSnapshot.status == .ready,
               healthSnapshot.workerRuntimeFeatureConfiguration?.configurationGeneration
                   == expectedGeneration {
                return;
            }
            if Date() >= readinessDeadline {
                throw WorkerReplacementJourneyFailure.workerAcknowledgementTimedOut;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    /// Writes one executable bash fixture that emits the idle event plus a
    /// runtime-policy acknowledgement for the given generation, then
    /// consumes stdin until the supervisor half-closes the command side.
    private static func writeFixtureScript(
        fixtureDirectoryUrl: URL,
        fixtureName: String,
        acknowledgedGeneration: String
    ) throws -> String {
        let runtimePolicyPayload: String =
            "{\"kind\":\"runtime_feature_configuration_applied\","
            + "\"worker_runtime_feature_configuration\":{\"configuration_generation\":"
            + "\"\(acknowledgedGeneration)\",\"persistent_prompt_cache_enabled\":true,"
            + "\"prompt_cache_maximum_size_bytes\":1073741824,\"loaded_model\":null}}";
        let scriptText: String = "#!/bin/bash\n"
            + FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: FakeWorkerEventEmitter.idleEventPayload())
            + FakeWorkerEventEmitter.emitLine(payload: runtimePolicyPayload)
            + "cat > /dev/null\n";
        let fixturePath: String = fixtureDirectoryUrl.appendingPathComponent(fixtureName).path;
        try scriptText.write(toFile: fixturePath, atomically: true, encoding: .utf8);
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: fixturePath);
        return fixturePath;
    }
}

/// Typed failures of the worker-replacement journey plumbing.
enum WorkerReplacementJourneyFailure: Error {

    case reloadRouteMissing;
    case generationMissing(String);
    case workerAcknowledgementTimedOut;
}

/// One admission journey: a chat executor that blocks between admission and
/// release, and the reload route bound to the same serving route table with
/// no worker control.
final class AdmissionReloadJourney {

    let delayedExecutor: DelayedAdmissionChatExecutor;
    let homeDirectoryUrl: URL;
    private let routeTable: RestRouteTable;

    private init(
        delayedExecutor: DelayedAdmissionChatExecutor,
        homeDirectoryUrl: URL,
        routeTable: RestRouteTable
    ) {
        self.delayedExecutor = delayedExecutor;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.routeTable = routeTable;
    }

    static func launch() throws -> AdmissionReloadJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-admission-reload-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        ConfigReloadJourney.writeRawConfigFile(homeDirectoryUrl, rawConfigJson: "{}");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: try RestChatJourneySupport.makeResolvedConfig().workerExecutablePath);
        let initialResolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        let delayedExecutor: DelayedAdmissionChatExecutor = DelayedAdmissionChatExecutor(
            modelId: RestChatJourneySupport.nonStreamingModelId);
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            chatContext: RestChatRouteContext(
                chatExecutor: delayedExecutor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: initialResolvedConfig,
                instancePaths: instancePaths),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: resolver,
                workerControl: nil,
                workerHealthState: WorkerHealthState(),
                generationActivityIdleProvider: { return !delayedExecutor.isAdmissionInFlight }));
        return AdmissionReloadJourney(
            delayedExecutor: delayedExecutor,
            homeDirectoryUrl: homeDirectoryUrl,
            routeTable: routeTable);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    /// Posts the blocking chat request on its own thread, mirroring the
    /// Rust journey's tokio::spawn of the oneshot request.
    func postChatOnThread() -> AdmissionReloadJourney.ThreadedOutcome {
        let outcome: AdmissionReloadJourney.ThreadedOutcome = AdmissionReloadJourney.ThreadedOutcome();
        let routeTable: RestRouteTable = self.routeTable;
        let chatThread: Thread = Thread {
            do {
                outcome.record(response: try RestChatJourneySupport.postChat(
                    routeTable: routeTable,
                    requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\","
                        + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}"));
            } catch {
                outcome.record(error: error);
            }
        };
        outcome.workerThread = chatThread;
        chatThread.start();
        return outcome;
    }

    func postConfigReloadOnThread() -> AdmissionReloadJourney.ThreadedOutcome {
        let outcome: AdmissionReloadJourney.ThreadedOutcome = AdmissionReloadJourney.ThreadedOutcome();
        let routeTable: RestRouteTable = self.routeTable;
        let reloadThread: Thread = Thread {
            do {
                let routeOutcome: RestRouteOutcome = routeTable.outcome(
                    method: RestConfigReloadEndpoint.routeMethod,
                    path: RestConfigReloadEndpoint.routePath);
                guard case .handler(let routeHandler) = routeOutcome else {
                    throw WorkerReplacementJourneyFailure.reloadRouteMissing;
                }
                outcome.record(response: try routeHandler(WorkerReplacementJourney.emptyRequest(
                    method: RestConfigReloadEndpoint.routeMethod,
                    path: RestConfigReloadEndpoint.routePath)));
            } catch {
                outcome.record(error: error);
            }
        };
        outcome.workerThread = reloadThread;
        reloadThread.start();
        return outcome;
    }

    func awaitAdmissionStarted(within timeoutSeconds: TimeInterval) throws -> Void {
        if self.delayedExecutor.admissionStartedSemaphore.wait(
            forTimeout: timeoutSeconds) == false {
            throw WorkerReplacementJourneyFailure.workerAcknowledgementTimedOut;
        }
    }

    func awaitCompletion(
        of outcome: AdmissionReloadJourney.ThreadedOutcome,
        within timeoutSeconds: TimeInterval
    ) throws -> Void {
        try outcome.join(within: timeoutSeconds);
    }

    func releaseAdmission() -> Void {
        self.delayedExecutor.releaseAdmissionSemaphore.signal();
    }
}

/// A chat executor that holds every admitted generation between its
/// admission signal and the journey's release, mirroring the Rust
/// DelayedAdmissionExecutor test double.
final class DelayedAdmissionChatExecutor: ChatGenerationExecuting, @unchecked Sendable {

    let admissionStartedSemaphore: BinarySemaphore;
    let releaseAdmissionSemaphore: BinarySemaphore;
    private let scriptedHealthSnapshot: WorkerHealthSnapshot;
    private let admissionStateLock: NSLock;
    private var admissionInFlight: Bool;

    init(modelId: String) {
        self.admissionStartedSemaphore = BinarySemaphore();
        self.releaseAdmissionSemaphore = BinarySemaphore();
        self.scriptedHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        self.admissionStateLock = NSLock();
        self.admissionInFlight = false;
    }

    var isAdmissionInFlight: Bool {
        self.admissionStateLock.lock();
        defer { self.admissionStateLock.unlock(); }
        return self.admissionInFlight;
    }

    func startChatGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent> {
        self.admissionStateLock.lock();
        self.admissionInFlight = true;
        self.admissionStateLock.unlock();
        self.admissionStartedSemaphore.signal();
        _ = self.releaseAdmissionSemaphore.wait(forTimeout: 10);
        self.admissionStateLock.lock();
        self.admissionInFlight = false;
        self.admissionStateLock.unlock();
        return [
            .completed(
                promptTokenCount: 1,
                generatedTokenCount: 1,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence),
        ];
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.scriptedHealthSnapshot;
    }
}

/// A one-shot semaphore over POSIX counters.
final class BinarySemaphore: @unchecked Sendable {

    private let dispatchSemaphore: DispatchSemaphore;

    init() {
        self.dispatchSemaphore = DispatchSemaphore(value: 0);
    }

    func signal() -> Void {
        _ = self.dispatchSemaphore.signal();
    }

    func wait(forTimeout timeoutSeconds: TimeInterval) -> Bool {
        let waitDeadline: Date = Date().addingTimeInterval(timeoutSeconds);
        let waitResult: DispatchTimeoutResult = self.dispatchSemaphore.wait(
            timeout: .now() + max(0, waitDeadline.timeIntervalSinceNow));
        return waitResult == .success;
    }
}

extension AdmissionReloadJourney {

    /// One request executing on its own thread with its outcome captured for
    /// the journey thread to inspect after a bounded join.
    final class ThreadedOutcome: @unchecked Sendable {

        var workerThread: Thread = Thread();
        private let outcomeLock: NSLock;
        private var recordedResponse: RestHttpResponse?;
        private var recordedError: Error?;

        init() {
            self.outcomeLock = NSLock();
        }

        var httpResponse: RestHttpResponse? {
            self.outcomeLock.lock();
            defer { self.outcomeLock.unlock(); }
            return self.recordedResponse;
        }

        func record(response: RestHttpResponse) -> Void {
            self.outcomeLock.lock();
            self.recordedResponse = response;
            self.outcomeLock.unlock();
        }

        func record(error: Error) -> Void {
            self.outcomeLock.lock();
            self.recordedError = error;
            self.outcomeLock.unlock();
        }

        /// Bounded join: the request must finish inside the window or the
        /// journey fails, proving the surface stayed responsive.
        func join(within timeoutSeconds: TimeInterval) throws -> Void {
            let joinDeadline: Date = Date().addingTimeInterval(timeoutSeconds);
            while Date() < joinDeadline {
                self.outcomeLock.lock();
                let hasFinished: Bool = self.recordedResponse != nil || self.recordedError != nil;
                self.outcomeLock.unlock();
                if hasFinished {
                    return;
                }
                Thread.sleep(forTimeInterval: 0.01);
            }
            throw WorkerReplacementJourneyFailure.workerAcknowledgementTimedOut;
        }
    }
}
