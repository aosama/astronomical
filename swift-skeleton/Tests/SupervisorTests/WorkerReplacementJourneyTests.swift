import Foundation;

import Testing;

import IpcProtocol;

@testable import Supervisor;

/**
 * Process-boundary acceptance journeys for transactional worker
 * replacement, migrating apps/supervisor/tests/hermetic/worker_replacement.rs:
 * the trusted worker survives every rejected candidate untouched, and a
 * committed swap happens only on the exact acknowledged pair of readiness
 * plus runtime configuration.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class TransactionalWorkerReplacementJourneyTests {

    /// A fully acknowledged candidate replaces the trusted worker and serves.
    @Test
    func should_replace_the_trusted_worker_only_after_exact_candidate_acknowledgement() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }

        let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration =
            try journey.supervisor.restartWorkerWithStartupConfiguration(
                candidateWorkerExecutablePath: journey.idleWorkerExecutablePath,
                candidateWorkerArguments: Array<String>(),
                candidateModelPolicyCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                    TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
                candidateStartupConfiguration: journey.startupConfiguration(
                    TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION));

        #expect(acknowledgedConfiguration.configurationGeneration
            == TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);
        let candidateHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();
        #expect(candidateHealth.status == .ready);
        #expect(
            candidateHealth.workerRuntimeFeatureConfiguration?.configurationGeneration
                == TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID);
    }

    /// Runtime-policy acknowledgement may precede the candidate's readiness.
    @Test
    func should_accept_candidate_configuration_before_initial_readiness() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }

        let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration =
            try journey.restartCandidate(
                modeArgument: "--replacement-candidate",
                candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                    TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
                configurationGeneration: TransactionalWorkerReplacementJourney.CONFIGURATION_BEFORE_READY_GENERATION);

        #expect(acknowledgedConfiguration.configurationGeneration
            == TransactionalWorkerReplacementJourney.CONFIGURATION_BEFORE_READY_GENERATION);
        #expect(journey.supervisor.workerHealthSnapshot().status == .ready);
    }

    /// The exact tagged FLUX runtime configuration commits a swap.
    @Test
    func should_match_the_exact_tagged_flux_runtime_configuration_during_replacement() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }

        let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration =
            try journey.restartCandidate(
                modeArgument: "--replacement-candidate",
                candidateCatalog: [
                    "FLUX.2-klein-4B": IdleWorkerJourneySupport.fluxRuntimeModelPolicy(
                        artifactRevision: "reviewed-revision"),
                ],
                configurationGeneration: TransactionalWorkerReplacementJourney.MATCHING_FLUX_GENERATION);

        #expect(acknowledgedConfiguration.configurationGeneration
            == TransactionalWorkerReplacementJourney.MATCHING_FLUX_GENERATION);
        #expect(journey.supervisor.workerHealthSnapshot().readyModelId == "FLUX.2-klein-4B");
    }

    /// A candidate acknowledging a different artifact revision is rejected.
    @Test
    func should_reject_flux_replacement_when_the_acknowledged_revision_differs() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--replacement-candidate",
            candidateCatalog: [
                "FLUX.2-klein-4B": IdleWorkerJourneySupport.fluxRuntimeModelPolicy(
                    artifactRevision: "different-revision"),
            ],
            configurationGeneration: TransactionalWorkerReplacementJourney.MATCHING_FLUX_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains(
                "disagrees with its acknowledged policy"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
    }

    /// A candidate executable that cannot start fails the replacement.
    @Test
    func should_keep_the_trusted_worker_when_candidate_launch_fails() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: nil,
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("failed to start worker process"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }

    /// A ready candidate that does not acknowledge its loaded model policy
    /// is rejected.
    @Test
    func should_reject_ready_candidate_without_matching_loaded_model_acknowledgement() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--replacement-candidate",
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.INCONSISTENT_READY_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("loaded model policy"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }

    /// A candidate acknowledging a foreign generation is rejected, reaped,
    /// and the trusted worker keeps serving.
    @Test
    func should_reap_mismatched_candidate_and_keep_trusted_worker_serving() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--replacement-candidate",
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("different configuration generation"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        IdleWorkerJourneySupport.assertCandidateWasReaped(journeyDirectoryPath: journey.journeyDirectoryPath);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }

    /// A candidate that never acknowledges is timed out and reaped.
    @Test
    func should_keep_the_trusted_worker_when_candidate_readiness_times_out() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 0.5);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--loading-forever",
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("candidate"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }

    /// A candidate that exits before acknowledging fails the replacement
    /// with process-exit diagnostics.
    @Test
    func should_keep_the_trusted_worker_when_candidate_exits_before_acknowledgement() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--mismatched-ready",
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.CANDIDATE_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("worker process exited"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }

    /// Generation-scoped events during the startup handshake are rejected.
    @Test
    func should_reject_generation_scoped_candidate_events_and_keep_trusted_worker() throws {
        let journey: TransactionalWorkerReplacementJourney = try TransactionalWorkerReplacementJourney.launch(modelLoadTimeout: 2);
        defer { journey.dispose() }
        let trustedHealth: WorkerHealthSnapshot = journey.supervisor.workerHealthSnapshot();

        let replacementError: Error = try journey.restartCandidateError(
            modeArgument: "--replacement-candidate",
            candidateCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                TransactionalWorkerReplacementJourney.CANDIDATE_MODEL_ID),
            configurationGeneration: TransactionalWorkerReplacementJourney.GENERATION_EVENT_GENERATION);

        #expect(
            WorkerControlError.describe(replacementError).contains("invalid startup event"),
            "unexpected replacement error: \(WorkerControlError.describe(replacementError))");
        journey.assertTrustedWorkerUnchanged(trustedHealth);
        try journey.assertGenerationSucceeds(TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID);
    }
}

/// One trusted idle fixture worker plus the candidate executable paths its
/// restart attempts launch, mirroring the Rust ReplacementTestContext.
final class TransactionalWorkerReplacementJourney {

    static let INITIAL_GENERATION: String = String(repeating: "1", count: 64);
    static let CANDIDATE_GENERATION: String = String(repeating: "2", count: 64);
    static let GENERATION_EVENT_GENERATION: String = String(repeating: "3", count: 64);
    static let CONFIGURATION_BEFORE_READY_GENERATION: String = String(repeating: "4", count: 64);
    static let INCONSISTENT_READY_GENERATION: String = String(repeating: "5", count: 64);
    static let MATCHING_FLUX_GENERATION: String = String(repeating: "6", count: 64);
    static let TRUSTED_MODEL_ID: String = "astronomical/trusted-model";
    static let CANDIDATE_MODEL_ID: String = "astronomical/candidate-model";

    let supervisor: WorkerSupervisor;
    let journeyDirectoryPath: String;
    let idleWorkerExecutablePath: String;
    private let journeyCleanup: () -> Void;

    private init(
        supervisor: WorkerSupervisor,
        journeyDirectoryPath: String,
        idleWorkerExecutablePath: String,
        journeyCleanup: @escaping () -> Void
    ) {
        self.supervisor = supervisor;
        self.journeyDirectoryPath = journeyDirectoryPath;
        self.idleWorkerExecutablePath = idleWorkerExecutablePath;
        self.journeyCleanup = journeyCleanup;
    }

    static func launch(modelLoadTimeout: TimeInterval) throws -> TransactionalWorkerReplacementJourney {
        let harness: IdleWorkerJourneySupport.IdleWorkerHarness =
            try IdleWorkerJourneySupport.launchConfiguredIdleWorker(
                modelPolicyCatalog: TransactionalWorkerReplacementJourney.candidateCatalog(
                    TransactionalWorkerReplacementJourney.TRUSTED_MODEL_ID),
                modelLoadTimeout: modelLoadTimeout,
                startupConfigurationBuilder: { (journeyDirectoryPath: String) -> WorkerStartupConfiguration? in
                    return TransactionalWorkerReplacementJourney.startupConfiguration(
                        TransactionalWorkerReplacementJourney.INITIAL_GENERATION,
                        loggingDirectory: journeyDirectoryPath)
                },
                workerArguments: [],
                cancellationAcknowledgementTimeout:
                    WorkerSupervisor.defaultCancellationAcknowledgementTimeoutSeconds);
        try IdleWorkerJourneySupport.waitForEffectiveGeneration(
            harness.supervisor,
            expectedGeneration: TransactionalWorkerReplacementJourney.INITIAL_GENERATION);
        return TransactionalWorkerReplacementJourney(
            supervisor: harness.supervisor,
            journeyDirectoryPath: harness.journeyDirectoryPath,
            idleWorkerExecutablePath: try IdleWorkerJourneySupport.locateBuiltExecutable(
                executableName: IdleWorkerJourneySupport.WORKER_EXECUTABLE_NAME),
            journeyCleanup: { () -> Void in
                harness.dispose();
            });
    }

    func dispose() -> Void {
        self.journeyCleanup();
    }

    /// The startup configuration with its logging and cache roots inside
    /// this journey's directory, mirroring the Rust helper.
    func startupConfiguration(_ configurationGeneration: String) -> WorkerStartupConfiguration {
        return TransactionalWorkerReplacementJourney.startupConfiguration(
            configurationGeneration,
            loggingDirectory: self.journeyDirectoryPath);
    }

    static func startupConfiguration(
        _ configurationGeneration: String,
        loggingDirectory: String
    ) -> WorkerStartupConfiguration {
        return WorkerStartupConfiguration(
            configurationGeneration: configurationGeneration,
            globalPromptCacheRootDirectory: loggingDirectory + "/cache",
            globalPromptCacheMaximumSizeBytes: 1_000_000_000,
            persistentPromptCacheEnabled: true,
            configuredMaximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: true,
            loggingDirectory: loggingDirectory,
            loggingLevel: .warn,
            retainedLogFileCount: 1);
    }

    /// Restarts through the fixture executable under one scenario mode.
    func restartCandidate(
        modeArgument: String,
        candidateCatalog: Dictionary<String, RuntimeModelPolicy>,
        configurationGeneration: String
    ) throws -> WorkerRuntimeFeatureConfiguration {
        return try self.supervisor.restartWorkerWithStartupConfiguration(
            candidateWorkerExecutablePath: self.idleWorkerExecutablePath,
            candidateWorkerArguments: [modeArgument],
            candidateModelPolicyCatalog: candidateCatalog,
            candidateStartupConfiguration: self.startupConfiguration(configurationGeneration));
    }
    /// The failing shape of restartCandidate, for the rejection journeys; a
    /// nil mode targets a missing executable.
    func restartCandidateError(
        modeArgument: String?,
        candidateCatalog: Dictionary<String, RuntimeModelPolicy>,
        configurationGeneration: String
    ) throws -> Error {
        let candidateExecutablePath: String = modeArgument == nil
            ? self.journeyDirectoryPath + "/missing-worker"
            : self.idleWorkerExecutablePath;
        do {
            _ = try self.supervisor.restartWorkerWithStartupConfiguration(
                candidateWorkerExecutablePath: candidateExecutablePath,
                candidateWorkerArguments: modeArgument == nil ? [] : [modeArgument!],
                candidateModelPolicyCatalog: candidateCatalog,
                candidateStartupConfiguration: self.startupConfiguration(configurationGeneration));
        } catch let replacementError {
            return replacementError;
        }
        throw TransactionalWorkerReplacementJourneyFailure.replacementUnexpectedlySucceeded;
    }

    func assertTrustedWorkerUnchanged(_ trustedHealth: WorkerHealthSnapshot) -> Void {
        #expect(self.supervisor.workerHealthSnapshot() == trustedHealth);
    }

    func assertGenerationSucceeds(_ modelId: String) throws -> Void {
        let streamEvents: Array<ChatGenerationStreamEvent> = try self.supervisor.startChatGeneration(
            TransactionalWorkerReplacementJourney.chatCommand(modelId: modelId));
        IdleWorkerJourneySupport.assertGenerationCompleted(
            streamEvents,
            generationLabel: "generation through \(modelId)");
    }

    /// The chat request shape every trusted-worker-still-serves assertion
    /// uses: the five-thousand-word Romeo and Juliet corpus.
    static func chatCommand(modelId: String) throws -> ChatGenerationCommand {
        let romeoAndJulietCorpus: String = try IdleWorkerJourneySupport.romeoAndJulietFiveThousandWords();
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: 1),
            model: modelId,
            messages: [.user(content: romeoAndJulietCorpus, images: [])],
            tools: [],
            toolChoice: .none,
            settings: ChatGenerationSettings(
                maxOutputTokens: 1,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }

    /// The single-model chat catalog the Rust journey builder constructs.
    static func candidateCatalog(_ modelId: String) -> Dictionary<String, RuntimeModelPolicy> {
        return [
            modelId: IdleWorkerJourneySupport.runtimeModelPolicy(
                modelId,
                modelDirectory: "/fictional/models/\(modelId)",
                maximumOutputTokens: 128),
        ];
    }
}

enum TransactionalWorkerReplacementJourneyFailure: Error {

    case replacementUnexpectedlySucceeded;
}
