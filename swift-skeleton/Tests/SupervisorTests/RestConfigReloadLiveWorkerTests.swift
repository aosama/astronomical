import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Config-reload and memory-ceiling journeys that drive a real fixture
 * worker process, migrating apps/supervisor/tests/rest_api/config_reload/
 * mixed_reload_configuration_generation.rs and the live-worker
 * maximum_mlx_memory.rs journeys: a mixed reload applies only the derived
 * memory configuration generation and the next model swap accepts it,
 * unrelated pending config changes reject a memory update, and a queued
 * reload memory setting rejected after a generation rolls the live state
 * back to the prior configuration.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestConfigReloadLiveWorkerTests {

    private static let mixedModelId: String = "astronomical/mixed-memory-reload-model";
    private static let delayedCompletionModelId: String = "astronomical/delayed-completion-model";
    static let romeoAndJulietPrompt: String =
        "You are a concise literature assistant. In one sentence, name the play "
        + "these lines come from: \"O Romeo, Romeo, wherefore art thou Romeo?\"";

    /// A reload that only changes the memory ceiling plus restart-required
    /// fields must apply the derived memory-only generation to the live
    /// worker, and the next model swap must acknowledge it.
    @Test
    func should_load_a_model_after_a_mixed_reload_applies_only_the_memory_configuration_generation() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: RestChatJourneySupport.makeResolvedConfig().workerStartupConfiguration(),
            modelPolicyCatalog: [
                RestConfigReloadLiveWorkerTests.mixedModelId:
                    ReloadLiveWorkerJourney.autoregressiveModelPolicy(
                        modelId: RestConfigReloadLiveWorkerTests.mixedModelId),
            ]);
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],"
                + "\"maximum_mlx_memory_gb\":32},\"diagnostics\":{\"log_level\":\"info\"}}");
        let memoryEffectiveGeneration: String = ResolvedConfigurationGeneration.deriveMemoryOnlyTransition(
            priorResolvedGeneration: journey.transitionState.currentReloadableConfig()
                .configurationGeneration,
            maximumMlxMemoryBytes: 32_000_000_000);
        journey.armMemoryAcknowledgement(
            effectiveMlxMemoryCeilingBytes: 32_000_000_000,
            configurationGeneration: memoryEffectiveGeneration);

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 200);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "restart_required");
        try WorkerReplacementJourney.waitForEffectiveGeneration(
            supervisor: journey.supervisor,
            expectedGeneration: memoryEffectiveGeneration);
        journey.armModelSwap(RestConfigReloadLiveWorkerTests.mixedModelId);
        let generationOutcome: ReloadLiveWorkerJourney.GenerationOutcome =
            journey.startGenerationOnThread(
                requestId: 9_002,
                modelId: RestConfigReloadLiveWorkerTests.mixedModelId);
        try ReloadLiveWorkerJourney.awaitReadyModel(
            supervisor: journey.supervisor,
            expectedModelId: RestConfigReloadLiveWorkerTests.mixedModelId);
        journey.pokeCompletion(requestId: 9_002);
        try generationOutcome.join(within: 5);
        let completionReason: ChatGenerationCompletionReason? = try generationOutcome
            .requireCompletionReason();
        #expect(completionReason == .endOfSequence);
    }

    /// A memory update must be rejected while unrelated configuration
    /// changes are pending, leaving the file and the live config untouched.
    @Test
    func should_require_full_reload_when_other_configuration_changes_are_pending() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: nil,
            modelPolicyCatalog: Dictionary());
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"diagnostics\":{\"log_level\":\"info\"}}");

        let memoryResponse: RestHttpResponse = try journey.putMaximumMlxMemory(
            maximumMlxMemoryGb: 32);

        #expect(memoryResponse.statusCode == 409);
        let configFileText: String = try String(
            contentsOf: journey.configFileUrl,
            encoding: .utf8);
        #expect(configFileText.contains("info"));
        #expect(journey.transitionState.currentReloadableConfig().maximumMlxMemoryBytes == nil);
    }

    /// A reload memory setting queued behind an active generation and then
    /// rejected by the worker must roll the live configuration back to the
    /// prior resolved generation while keeping the persisted candidate.
    @Test
    func should_rollback_live_state_when_a_reloaded_memory_setting_is_rejected_after_queueing() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: nil,
            modelPolicyCatalog: [
                RestConfigReloadLiveWorkerTests.delayedCompletionModelId:
                    ReloadLiveWorkerJourney.autoregressiveModelPolicy(
                        modelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId),
            ]);
        defer { journey.dispose() }
        let initialGeneration: String = journey.transitionState.currentReloadableConfig()
            .configurationGeneration;
        journey.armModelSwap(RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        let generationOutcome: ReloadLiveWorkerJourney.GenerationOutcome =
            journey.startGenerationOnThread(
                requestId: 9_001,
                modelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        try ReloadLiveWorkerJourney.awaitReadyModel(
            supervisor: journey.supervisor,
            expectedModelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],"
                + "\"maximum_mlx_memory_gb\":31}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();
        #expect(reloadResponse.statusCode == 200);
        let conflictingMemoryResponse: RestHttpResponse = try journey.putMaximumMlxMemory(
            maximumMlxMemoryGb: 32);
        #expect(conflictingMemoryResponse.statusCode == 409);

        journey.armMemoryRejection(requestedMlxMemoryCeilingBytes: 31_000_000_000);
        journey.pokeCompletion(requestId: 9_001);
        try generationOutcome.join(within: 5);
        #expect(try generationOutcome.requireCompletionReason() == .endOfSequence);
        let rollbackDeadline: Date = Date().addingTimeInterval(5);
        while journey.transitionState.currentReloadableConfig().configurationGeneration
            != initialGeneration {
            if Date() >= rollbackDeadline {
                Issue.record(
                    "the reloadable generation never rolled back to \(initialGeneration)");
                break;
            }
            Thread.sleep(forTimeInterval: 0.025);
        }
        let configFileText: String = try String(
            contentsOf: journey.configFileUrl,
            encoding: .utf8);
        #expect(configFileText.contains("31"));
    }
}
