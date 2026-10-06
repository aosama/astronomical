import Foundation;

import AstronomicalConfig;
import IpcProtocol;

@testable import Supervisor;

/**
 * Shared fixture and raw-HTTP helpers for the /v1/status acceptance
 * journeys: one resolved-configuration fixture builder with knobs for the
 * diagnostics, unmatched identities, and memory ceilings, plus the loopback
 * exchange and JSON-envelope accessors every journey uses. The temporary
 * configuration state lives only for the support instance's lifetime.
 */
final class RestStatusJourneySupport {

    private var temporaryRootPath: String?;

    func makeResolvedConfig(resolvedGeneration: String) throws -> ResolvedRuntimeConfig {
        let discoveredModels: Array<DiscoveryDiscoveredModel> = [
            DiscoveryDiscoveredModel(
                modelId: "synthetic-chat-model",
                providerModelId: nil,
                modelFamily: .modernbert,
                revision: "0000000000000000000000000000000000000000",
                modelDirectory: FilePath(string: "/models/synthetic-chat-model"),
                capabilities: .chat(DiscoveryChatModelCapabilities(
                    contextWindowTokens: 4_096,
                    maximumInputTokens: 4_095,
                    maximumOutputTokens: 1_024,
                    supportsVision: false,
                    supportsReasoning: false,
                    supportsToolCalls: false)),
                license: nil,
                modelSizeBytes: 400_000_000),
        ];
        let emptyUserConfig: AstronomicalConfig = try self.emptyConfig();
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            modelPolicyCatalog[discoveredModel.modelId] = try ResolvedModelPolicyCatalog.resolve(
                userConfig: emptyUserConfig,
                discoveredModels: discoveredModels,
                artifactContextWindows: Dictionary<String, UInt32>())[discoveredModel.modelId];
        }
        return ResolvedRuntimeConfig(
            configurationGeneration: resolvedGeneration,
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/astronomical-inference-worker"),
            discoveredModels: discoveredModels,
            modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>(),
            configuredModelDirectories: Array<FilePath>(),
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>(),
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            experimentalQwenThinkingChannelSeedEnabled: false,
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: 50_000_000_000,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/state/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:0",
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            loggingConfig: LoggingConfig(
                directory: FilePath(string: "/state/logs"),
                level: LogLevel.warn,
                retainedFiles: 7));
    }

    func exchangeObject(port: UInt16, requestTarget: String) throws -> (Int, [String: Any]) {
        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: port,
            requestText: "GET \(requestTarget) HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let bodyText: String = try self.requireResponseText(responseText);
        guard let statusToken: Substring = bodyText.split(separator: " ", maxSplits: 2).dropFirst().first,
            let statusCode: Int = Int(statusToken) else {
            throw RestStatusTestFailure.malformedStatusLine;
        }
        guard let bodyStart: String.Index = bodyText.range(of: "\r\n\r\n")?.upperBound else {
            throw RestStatusTestFailure.missingResponseBody;
        }
        let bodyJsonText: String = String(bodyText[bodyStart...]);
        guard let envelopeObject: [String: Any] = try JSONSerialization.jsonObject(
            with: Data(bodyJsonText.utf8)) as? [String: Any] else {
            throw RestStatusTestFailure.nonJsonEnvelope;
        }
        return (statusCode, envelopeObject);
    }

    func requireObject(_ anyValue: Any?, failure: RestStatusTestFailure) throws -> [String: Any] {
        guard let objectValue: [String: Any] = anyValue as? [String: Any] else {
            throw failure;
        }
        return objectValue;
    }

    func requireArray(_ anyValue: Any?, failure: RestStatusTestFailure) throws -> Array<Any> {
        guard let arrayValue: Array<Any> = anyValue as? Array<Any> else {
            throw failure;
        }
        return arrayValue;
    }

    func requireResponseText(_ responseText: String?) throws -> String {
        guard let unwrappedResponseText: String = responseText else {
            throw RestStatusTestFailure.missingResponse;
        }
        return unwrappedResponseText;
    }

    private func emptyConfig() throws -> AstronomicalConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "arests-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        self.temporaryRootPath = temporaryStateDirectory;
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        return try AstronomicalConfig.loadFromInstancePaths(instancePaths);
    }

    deinit {
        if let temporaryRootPath: String = self.temporaryRootPath {
            try? FileManager.default.removeItem(atPath: temporaryRootPath);
        }
    }
}

enum RestStatusTestFailure: Error {
    case missingResponse;
    case malformedStatusLine;
    case missingResponseBody;
    case nonJsonEnvelope;
    case missingApplicationSection;
    case missingConfigurationSection;
    case missingWorkerAcknowledgement;
    case missingReadyModelSummary;
    case missingPromptCacheSummary;
    case missingMemorySummary;
    case missingChunkingSummary;
    case missingExpertResidency;
    case missingMlxSnapshot;
    case missingDiagnostics;
    case missingConfigurationTriple;
}
