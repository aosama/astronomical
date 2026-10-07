import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Application-layer setting application over the scripted chat surface,
 * migrating apps/supervisor/tests/rest_api/config_reload/output_limits.rs
 * and qwen_thinking_channel_seed.rs: each model's generation defaults fill
 * omitted chat and responses settings while explicit request values win, and
 * the config-gated Qwen thinking-channel seed reaches both surfaces only for
 * a discovered Qwen3.5 model with the experimental flag enabled.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestModelDefaultsAndSeedTests {

    static let primaryModelId: String = "astronomical/application-test-model";
    static let secondaryModelId: String = "organization/secondary-model";
    static let romeoAndJulietThinkingSeed: String =
        "Two households, both alike in dignity, in fair Verona.";

    @Test
    func should_apply_each_models_defaults_to_omitted_chat_settings() throws {
        let journey: ModelDefaultsJourney = try ModelDefaultsJourney.launch([
            (
                RestModelDefaultsAndSeedTests.primaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 4_000, temperatureThousandths: 700, topPThousandths: 900)),
            (
                RestModelDefaultsAndSeedTests.secondaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 6_000, temperatureThousandths: 200, topPThousandths: 800)),
        ]);

        for modelId: String in [
            RestModelDefaultsAndSeedTests.primaryModelId,
            RestModelDefaultsAndSeedTests.secondaryModelId,
        ] {
            let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: journey.routeTable,
                requestBody: "{\"model\":\"\(modelId)\",\"messages\":[{\"role\":\"user\","
                    + "\"content\":\"hello\"}],\"stream\":true}");
            #expect(chatResponse.statusCode == 200);
        }

        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 2);
        #expect(receivedCommands[0].settings.maxOutputTokens == 4_000);
        #expect(receivedCommands[0].settings.temperatureThousandths == 700);
        #expect(receivedCommands[0].settings.topPThousandths == 900);
        #expect(receivedCommands[1].settings.maxOutputTokens == 6_000);
        #expect(receivedCommands[1].settings.temperatureThousandths == 200);
        #expect(receivedCommands[1].settings.topPThousandths == 800);
    }

    @Test
    func should_keep_explicit_chat_settings_above_the_models_output_default() throws {
        let journey: ModelDefaultsJourney = try ModelDefaultsJourney.launch([
            (
                RestModelDefaultsAndSeedTests.primaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 128, temperatureThousandths: 700, topPThousandths: 900)),
        ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: journey.routeTable,
            requestBody: "{\"model\":\"\(RestModelDefaultsAndSeedTests.primaryModelId)\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
                + "\"max_tokens\":20000,\"temperature\":0.3,\"top_p\":0.6,\"stream\":true}");

        #expect(chatResponse.statusCode == 200);
        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 1);
        #expect(receivedCommands[0].settings.maxOutputTokens == 20_000);
        #expect(receivedCommands[0].settings.temperatureThousandths == 300);
        #expect(receivedCommands[0].settings.topPThousandths == 600);
    }

    @Test
    func should_apply_each_models_defaults_to_omitted_responses_settings() throws {
        let journey: ModelDefaultsJourney = try ModelDefaultsJourney.launch([
            (
                RestModelDefaultsAndSeedTests.primaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 4_000, temperatureThousandths: 700, topPThousandths: 900)),
            (
                RestModelDefaultsAndSeedTests.secondaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 6_000, temperatureThousandths: 200, topPThousandths: 800)),
        ]);

        for modelId: String in [
            RestModelDefaultsAndSeedTests.primaryModelId,
            RestModelDefaultsAndSeedTests.secondaryModelId,
        ] {
            let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
                routeTable: journey.routeTable,
                requestBody: "{\"model\":\"\(modelId)\",\"input\":\"hello\",\"stream\":true}");
            #expect(responsesResponse.statusCode == 200);
        }

        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 2);
        #expect(receivedCommands[0].settings.maxOutputTokens == 4_000);
        #expect(receivedCommands[0].settings.temperatureThousandths == 700);
        #expect(receivedCommands[0].settings.topPThousandths == 900);
        #expect(receivedCommands[1].settings.maxOutputTokens == 6_000);
        #expect(receivedCommands[1].settings.temperatureThousandths == 200);
        #expect(receivedCommands[1].settings.topPThousandths == 800);
    }

    @Test
    func should_keep_explicit_responses_settings_above_the_models_output_default() throws {
        let journey: ModelDefaultsJourney = try ModelDefaultsJourney.launch([
            (
                RestModelDefaultsAndSeedTests.primaryModelId,
                ModelDefaultsJourney.generationDefaults(
                    maximumOutputTokens: 128, temperatureThousandths: 700, topPThousandths: 900)),
        ]);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: journey.routeTable,
            requestBody: "{\"model\":\"\(RestModelDefaultsAndSeedTests.primaryModelId)\","
                + "\"input\":\"hello\",\"max_output_tokens\":20000,\"temperature\":0.3,"
                + "\"top_p\":0.6,\"stream\":true}");

        #expect(responsesResponse.statusCode == 200);
        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 1);
        #expect(receivedCommands[0].settings.maxOutputTokens == 20_000);
        #expect(receivedCommands[0].settings.temperatureThousandths == 300);
        #expect(receivedCommands[0].settings.topPThousandths == 600);
    }

    @Test
    func should_guard_both_chat_surfaces_with_the_experimental_thinking_seed_flag() throws {
        let seedGuardCases: Array<(Bool, ModelFamily, String?)> = [
            (false, ModelFamily.qwen35, nil),
            (true, ModelFamily.qwen35, RestModelDefaultsAndSeedTests.romeoAndJulietThinkingSeed),
            (true, ModelFamily.k2HorizonMova, nil),
        ];
        for (thinkingChannelSeedEnabled, modelFamily, expectedSeed) in seedGuardCases {
            let journey: ModelDefaultsJourney = try ModelDefaultsJourney.launch(
                modelDefaults: [(
                    RestModelDefaultsAndSeedTests.primaryModelId,
                    ModelDefaultsJourney.generationDefaults(
                        maximumOutputTokens: 128,
                        temperatureThousandths: nil,
                        topPThousandths: nil))],
                thinkingChannelSeedEnabled: thinkingChannelSeedEnabled,
                thinkingSeedModelFamily: modelFamily);
            defer { journey.dispose() }

            let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: journey.routeTable,
                requestBody: "{\"model\":\"\(RestModelDefaultsAndSeedTests.primaryModelId)\","
                    + "\"messages\":[{\"role\":\"user\",\"content\":\"Romeo and Juliet\"}],"
                    + "\"stream\":true}");
            let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
                routeTable: journey.routeTable,
                requestBody: "{\"model\":\"\(RestModelDefaultsAndSeedTests.primaryModelId)\","
                    + "\"input\":\"Romeo and Juliet\",\"stream\":true}");
            #expect(chatResponse.statusCode == 200);
            #expect(responsesResponse.statusCode == 200);

            let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
            #expect(receivedCommands.count == 2);
            for generationCommand: ChatGenerationCommand in receivedCommands {
                #expect(generationCommand.qwenThinkingChannelSeed == expectedSeed);
            }
        }
    }
}

/// One application-defaults journey: a resolved configuration advertising
/// the scripted models with their generation defaults, a recording chat
/// executor, and one serving route table carrying both answer surfaces.
final class ModelDefaultsJourney {

    let executor: ScriptedChatExecutor;
    let routeTable: RestRouteTable;
    let homeDirectoryUrl: URL;

    private init(
        executor: ScriptedChatExecutor,
        routeTable: RestRouteTable,
        homeDirectoryUrl: URL
    ) {
        self.executor = executor;
        self.routeTable = routeTable;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch(
        _ modelDefaults: Array<(String, RuntimeModelGenerationDefaults)>
    ) throws -> ModelDefaultsJourney {
        return try ModelDefaultsJourney.launch(
            modelDefaults: modelDefaults,
            thinkingChannelSeedEnabled: false,
            thinkingSeedModelFamily: .qwen35);
    }

    static func launch(
        modelDefaults: Array<(String, RuntimeModelGenerationDefaults)>,
        thinkingChannelSeedEnabled: Bool,
        thinkingSeedModelFamily: ModelFamily
    ) throws -> ModelDefaultsJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-model-defaults-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        try "\(RestModelDefaultsAndSeedTests.romeoAndJulietThinkingSeed)\n".write(
            to: URL(fileURLWithPath: instancePaths.qwenThinkingChannelSeedFilePath.string),
            atomically: true,
            encoding: .utf8);
        var resolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        resolvedConfig.experimentalQwenThinkingChannelSeedEnabled = thinkingChannelSeedEnabled;
        resolvedConfig.discoveredModels = modelDefaults.map({ (modelDefault: (String, RuntimeModelGenerationDefaults)) -> DiscoveryDiscoveredModel in
            return ModelDefaultsJourney.discoveredChatModel(
                modelId: modelDefault.0,
                modelFamily: thinkingSeedModelFamily);
        });
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        for (modelId, generationDefaults): (String, RuntimeModelGenerationDefaults) in modelDefaults {
            modelPolicyCatalog[modelId] = ModelDefaultsJourney.modelPolicy(
                modelId: modelId,
                generationDefaults: generationDefaults);
        }
        resolvedConfig.modelPolicyCatalog = modelPolicyCatalog;
        let executor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestModelDefaultsAndSeedTests.primaryModelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()),
            streamEvents: [
                .completed(
                    promptTokenCount: 1,
                    generatedTokenCount: 1,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            chatContext: RestChatRouteContext(
                chatExecutor: executor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: resolvedConfig,
                instancePaths: instancePaths),
            responsesContext: RestResponsesRouteContext(
                responsesExecutor: executor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: resolvedConfig,
                instancePaths: instancePaths));
        return ModelDefaultsJourney(
            executor: executor,
            routeTable: routeTable,
            homeDirectoryUrl: homeDirectoryUrl);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    static func generationDefaults(
        maximumOutputTokens: UInt16,
        temperatureThousandths: UInt16?,
        topPThousandths: UInt16?
    ) -> RuntimeModelGenerationDefaults {
        return RuntimeModelGenerationDefaults(
            maximumOutputTokens: maximumOutputTokens,
            configuredMaximumOutputTokens: nil,
            temperatureThousandths: temperatureThousandths,
            topPThousandths: topPThousandths);
    }

    private static func modelPolicy(
        modelId: String,
        generationDefaults: RuntimeModelGenerationDefaults
    ) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            generationDefaults: generationDefaults,
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 8_192,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: WorkerModelConfiguration.autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: modelId,
                    maximumContextTokens: 4_096,
                    maximumOutputTokens: 1_024,
                    chunking: WorkerChunkingConfiguration(
                        fixedPromptProcessingChunkSizeTokens: 512,
                        fixedSsdStreamingPromptProcessingChunkSizeTokens: 512,
                        fullAttentionKeyValueGrowthTokens: 512,
                        prefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                        promptCacheBlockTokens: nil,
                        promptCacheCommonPrefixStrideBlocks: 1,
                        experimentalDecodeStageAttributionEnabled: false,
                        experimentalQuantizedKvCacheEnabled: false,
                        experimentalFusedMoeDecodeEnabled: false))));
    }

    private static func discoveredChatModel(
        modelId: String,
        modelFamily: ModelFamily
    ) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: nil,
            modelFamily: modelFamily,
            revision: "test-revision",
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 262_144,
                maximumInputTokens: 262_143,
                maximumOutputTokens: 65_535,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 1);
    }
}
