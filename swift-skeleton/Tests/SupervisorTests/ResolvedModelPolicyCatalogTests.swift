import Foundation;
import XCTest;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/**
 * Acceptance journeys for the runtime policy catalog: one discovered model
 * becomes the exact immutable execution policy the worker receives, for the
 * chat, image, and embedding families, with config inheritance applied on
 * top of the artifact capability.
 */
final class ResolvedModelPolicyCatalogTests: XCTestCase {

    private static let CHAT_CAPABILITIES: DiscoveryChatModelCapabilities = DiscoveryChatModelCapabilities(
        contextWindowTokens: 32_768,
        maximumInputTokens: 32_767,
        maximumOutputTokens: 8_192,
        supportsVision: false,
        supportsReasoning: true,
        supportsToolCalls: true);

    private static func chatModel(
        modelId: String,
        modelFamily: ModelFamily,
        capabilities: DiscoveryChatModelCapabilities
    ) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: nil,
            modelFamily: modelFamily,
            revision: "main",
            modelDirectory: FilePath(string: "/models/qwen3"),
            capabilities: .chat(capabilities),
            license: nil,
            modelSizeBytes: 4_000_000_000);
    }

    private var temporaryStateDirectory: String?;

    override func tearDown() {
        if let temporaryStateDirectory: String = self.temporaryStateDirectory {
            try? FileManager.default.removeItem(atPath: temporaryStateDirectory);
            self.temporaryStateDirectory = nil;
        }
        super.tearDown();
    }

    private static func imageCapabilities(defaultSteps: UInt16) -> DiscoveryImageGenerationCapabilities {
        return DiscoveryImageGenerationCapabilities(
            supportsTextToImage: true,
            supportsImageEditing: false,
            supportsMultipleReferenceImages: false,
            defaultSteps: defaultSteps,
            minimumDimensionPixels: 256,
            maximumDimensionPixels: 4_096,
            dimensionMultiplePixels: 64);
    }

    private func loadConfig(contents: String) throws -> AstronomicalConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "asup-catalog-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        self.temporaryStateDirectory = temporaryStateDirectory;
        let developmentPaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: developmentPaths.stateDirectory.string,
            withIntermediateDirectories: true);
        let configFileUrl: URL = URL(fileURLWithPath: developmentPaths.configFilePath.string);
        try contents.write(to: configFileUrl, atomically: true, encoding: String.Encoding.utf8);
        return try AstronomicalConfig.loadFromInstancePaths(developmentPaths);
    }

    func testChatPolicyCarriesCapabilityAndInternalRequestDefaults() throws {
        let userConfig: AstronomicalConfig = try self.loadConfig(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}");
        let discoveredModel: DiscoveryDiscoveredModel = ResolvedModelPolicyCatalogTests.chatModel(
            modelId: "qwen3-model",
            modelFamily: .qwen35,
            capabilities: ResolvedModelPolicyCatalogTests.CHAT_CAPABILITIES);

        let modelPolicies: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: [discoveredModel],
            // The pre-config artifact window stays the routing default even
            // when the capability advertises a different geometry.
            artifactContextWindows: ["qwen3-model": 30_000]);

        let modelPolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["qwen3-model"]);
        XCTAssertEqual(modelPolicy.modelDirectory, FilePath(string: "/models/qwen3"));
        XCTAssertNil(modelPolicy.configuredMaximumContextTokens);
        XCTAssertEqual(modelPolicy.defaultMaximumContextTokens, 30_000);
        XCTAssertEqual(modelPolicy.generationDefaults.maximumOutputTokens, UInt16(ResolvedModelConfig.defaultMaximumOutputTokens));
        XCTAssertNil(modelPolicy.generationDefaults.temperatureThousandths);
        guard case let .autoregressive(workerConfiguration) = modelPolicy.workerModelConfiguration else {
            return XCTFail("expected an autoregressive worker policy, got \(modelPolicy.workerModelConfiguration)");
        }
        XCTAssertEqual(workerConfiguration.modelId, "qwen3-model");
        XCTAssertEqual(workerConfiguration.maximumContextTokens, 32_768);
        XCTAssertEqual(workerConfiguration.maximumOutputTokens, 8_192);
        XCTAssertEqual(
            workerConfiguration.chunking,
            RuntimeModelPolicy.workerChunkingConfiguration(from: ChunkingConfig.defaultConfig()));
    }

    func testChatPolicyAppliesConfiguredInheritanceAndThousandths() throws {
        let userConfig: AstronomicalConfig = try self.loadConfig(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]},\"models\":{\"qwen3-model\":{"
                + "\"limits\":{\"maximum_context_tokens\":16384},"
                + "\"generation_defaults\":{\"maximum_output_tokens\":4096,\"temperature\":0.5,\"top_p\":0.9}}}}");
        let discoveredModel: DiscoveryDiscoveredModel = ResolvedModelPolicyCatalogTests.chatModel(
            modelId: "qwen3-model",
            modelFamily: .qwen35,
            capabilities: ResolvedModelPolicyCatalogTests.CHAT_CAPABILITIES);

        let modelPolicies: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: [discoveredModel],
            artifactContextWindows: Dictionary<String, UInt32>());

        let modelPolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["qwen3-model"]);
        XCTAssertEqual(modelPolicy.configuredMaximumContextTokens, 16_384);
        // No artifact window entry: the capability window is the default.
        XCTAssertEqual(modelPolicy.defaultMaximumContextTokens, 32_768);
        XCTAssertEqual(modelPolicy.generationDefaults.maximumOutputTokens, 4_096);
        XCTAssertEqual(modelPolicy.generationDefaults.configuredMaximumOutputTokens, 4_096);
        XCTAssertEqual(modelPolicy.generationDefaults.temperatureThousandths, 500);
        XCTAssertEqual(modelPolicy.generationDefaults.topPThousandths, 900);
    }

    func testImagePoliciesStayTypedToTheirDiscoveredFamily() throws {
        let userConfig: AstronomicalConfig = try self.loadConfig(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}");
        let fluxModel: DiscoveryDiscoveredModel = DiscoveryDiscoveredModel(
            modelId: "flux-model",
            providerModelId: nil,
            modelFamily: .flux2Klein,
            revision: "step-8",
            modelDirectory: FilePath(string: "/models/flux"),
            capabilities: .imageGeneration(ResolvedModelPolicyCatalogTests.imageCapabilities(defaultSteps: 8)),
            license: nil,
            modelSizeBytes: 8_000_000_000);
        let qwenImageModel: DiscoveryDiscoveredModel = DiscoveryDiscoveredModel(
            modelId: "qwen-image-model",
            providerModelId: nil,
            modelFamily: .qwenImage21,
            revision: "main",
            modelDirectory: FilePath(string: "/models/qwen-image"),
            capabilities: .imageGeneration(ResolvedModelPolicyCatalogTests.imageCapabilities(defaultSteps: 20)),
            license: nil,
            modelSizeBytes: 9_000_000_000);

        let modelPolicies: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: [fluxModel, qwenImageModel],
            artifactContextWindows: Dictionary<String, UInt32>());

        let fluxPolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["flux-model"]);
        guard case let .flux2Klein(fluxConfiguration) = fluxPolicy.workerModelConfiguration else {
            return XCTFail("expected a flux worker policy, got \(fluxPolicy.workerModelConfiguration)");
        }
        XCTAssertEqual(fluxConfiguration.artifactRevision, "step-8");
        XCTAssertEqual(fluxPolicy.generationDefaults, RuntimeModelGenerationDefaults.inert());

        let qwenImagePolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["qwen-image-model"]);
        guard case .qwenImage21 = qwenImagePolicy.workerModelConfiguration else {
            return XCTFail("expected a qwen-image worker policy, got \(qwenImagePolicy.workerModelConfiguration)");
        }
    }

    func testImageCapabilityOnAChatFamilyFallsBackToTheFluxIdentity() throws {
        let userConfig: AstronomicalConfig = try self.loadConfig(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}");
        // A discovered directory that classifies as a chat family but
        // advertises an image capability receives the flux identity it never
        // verified, so worker selection fails closed downstream.
        let misclassifiedModel: DiscoveryDiscoveredModel = DiscoveryDiscoveredModel(
            modelId: "misclassified-model",
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: "main",
            modelDirectory: FilePath(string: "/models/misclassified"),
            capabilities: .imageGeneration(ResolvedModelPolicyCatalogTests.imageCapabilities(defaultSteps: 4)),
            license: nil,
            modelSizeBytes: 1_000_000_000);

        let modelPolicies: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: [misclassifiedModel],
            artifactContextWindows: Dictionary<String, UInt32>());

        let modelPolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["misclassified-model"]);
        guard case .flux2Klein = modelPolicy.workerModelConfiguration else {
            return XCTFail("expected the flux fallback, got \(modelPolicy.workerModelConfiguration)");
        }
    }

    func testEmbeddingsPolicyCarriesGeometryAndStaysModernbert() throws {
        let userConfig: AstronomicalConfig = try self.loadConfig(
            contents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,\"runtime\":{\"model_directories\":[]}}");
        let embeddingModel: DiscoveryDiscoveredModel = DiscoveryDiscoveredModel(
            modelId: "modernbert-model",
            providerModelId: nil,
            modelFamily: .modernbert,
            revision: "main",
            modelDirectory: FilePath(string: "/models/modernbert"),
            capabilities: .embeddings(DiscoveryEmbeddingModelCapabilities(
                vectorWidth: 1_024,
                maximumInputTokens: 8_192)),
            license: nil,
            modelSizeBytes: 500_000_000);

        let modelPolicies: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: [embeddingModel],
            artifactContextWindows: Dictionary<String, UInt32>());

        let modelPolicy: RuntimeModelPolicy = try XCTUnwrap(modelPolicies["modernbert-model"]);
        guard case let .embeddings(embeddingConfiguration) = modelPolicy.workerModelConfiguration else {
            return XCTFail("expected an embeddings worker policy, got \(modelPolicy.workerModelConfiguration)");
        }
        XCTAssertEqual(embeddingConfiguration.modelId, "modernbert-model");
        XCTAssertEqual(embeddingConfiguration.modelFamily, WorkerEmbeddingModelFamily.modernBert);
        XCTAssertEqual(embeddingConfiguration.vectorWidth, 1_024);
        XCTAssertEqual(embeddingConfiguration.maximumInputTokens, 8_192);
        XCTAssertEqual(embeddingConfiguration.artifactRevision, "main");
        XCTAssertEqual(modelPolicy.generationDefaults, RuntimeModelGenerationDefaults.inert());
    }
}
