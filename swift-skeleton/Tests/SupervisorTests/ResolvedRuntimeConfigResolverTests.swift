import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for runtime resolution: an operator's instance config
 * (authored config.json plus model directories) resolves into one snapshot
 * the daemon serves and names path-free — discovered models join the policy
 * catalog, configured-but-absent identities stay unmatched, memory and
 * cache preferences carry through to the worker bootstrap, and the resolved
 * generation is stable across loads. The journey also proves the automatic
 * Library root outranks authored roots for the same identity.
 */
@Suite(.tags(.hermeticJourney))
final class ResolvedRuntimeConfigResolverTests {

    private var temporaryRootPath: String?;

    deinit {
        if let temporaryRootPath: String = self.temporaryRootPath {
            try? FileManager.default.removeItem(atPath: temporaryRootPath);
        }
    }

    private func makeTemporaryRoot() throws -> String {
        let temporaryRootPath: String = NSTemporaryDirectory() + "aresv-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryRootPath, withIntermediateDirectories: true);
        self.temporaryRootPath = temporaryRootPath;
        return temporaryRootPath;
    }

    /** A minimal executable ModernBERT artifact with the reviewed 8-bit profile. */
    private func writeModernbertModel(modelId: String, modelsRootPath: String) throws -> Void {
        let modelDirectoryPath: String = modelsRootPath + "/" + modelId;
        try FileManager.default.createDirectory(atPath: modelDirectoryPath, withIntermediateDirectories: true);
        let configJson: String = "{\"model_type\":\"modernbert\",\"hidden_size\":768,"
            + "\"max_position_embeddings\":8192,\"quantization\":{\"bits\":8}}";
        try configJson.write(
            to: URL(fileURLWithPath: modelDirectoryPath + "/config.json"),
            atomically: true,
            encoding: String.Encoding.utf8);
        try Data([0x01, 0x02, 0x03]).write(to: URL(fileURLWithPath: modelDirectoryPath + "/model.safetensors"));
        try "{}".write(
            to: URL(fileURLWithPath: modelDirectoryPath + "/tokenizer.json"),
            atomically: true,
            encoding: String.Encoding.utf8);
    }

    private func makeResolver(
        stateDirectoryPath: String,
        configContents: String
    ) throws -> ResolvedRuntimeConfigResolver {
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: stateDirectoryPath),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        try configContents.write(
            to: URL(fileURLWithPath: instancePaths.configFilePath.string),
            atomically: true,
            encoding: String.Encoding.utf8);
        return ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: FilePath(string: "/opt/astronomical/bin/inference-worker"));
    }

    @Test
    func should_resolve_an_operator_config_into_the_complete_snapshot() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let modelsRootPath: String = temporaryRootPath + "/authored-models";
        try self.writeModernbertModel(modelId: "EmbeddedBert", modelsRootPath: modelsRootPath);
        let configContents: String = "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
            + "\"runtime\":{\"model_directories\":[\"\(modelsRootPath)\"],"
            + "\"maximum_mlx_memory_gb\":24,\"default_model\":\"GhostModel\"},"
            + "\"models\":{\"GhostModel\":{\"limits\":{\"maximum_context_tokens\":1024}}}}";
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: temporaryRootPath + "/state",
            configContents: configContents);

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        let reloadedConfig: ResolvedRuntimeConfig = try resolver.load();

        #expect(resolvedConfig == reloadedConfig);
        #expect(resolvedConfig.configurationGeneration.count == 64);
        #expect(resolvedConfig.workerExecutablePath == FilePath(string: "/opt/astronomical/bin/inference-worker"));
        #expect(resolvedConfig.discoveredModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        } == ["EmbeddedBert"]);
        #expect(resolvedConfig.unmatchedModelConfigIds == ["GhostModel"]);
        #expect(resolvedConfig.maximumMlxMemoryBytes == 24_000_000_000);
        // A non-standard (temporary) state directory binds the loopback
        // placeholder; only the standard instance directories carry the
        // fixed channel ports.
        #expect(resolvedConfig.bindAddress.hasPrefix("127.0.0.1:"), "bind address should stay loopback: \(resolvedConfig.bindAddress)");
        #expect(
            resolvedConfig.promptCacheConfig.globalPromptCacheRootDirectory
                == resolver.resolvedInstancePaths.promptCacheDirectory);

        // The discovered model reaches the catalog with embeddings policy.
        let modelPolicy: RuntimeModelPolicy? = resolvedConfig.modelPolicyCatalog["EmbeddedBert"];
        guard case .embeddings = modelPolicy?.workerModelConfiguration else {
            Issue.record(Comment(stringLiteral: "expected an embeddings worker policy, got \(String(describing: modelPolicy?.workerModelConfiguration))"));
            return;
        }

        // The bootstrap DTO carries the resolved worker settings.
        let workerStartupConfiguration: WorkerStartupConfiguration = resolvedConfig.workerStartupConfiguration();
        #expect(workerStartupConfiguration.configurationGeneration == resolvedConfig.configurationGeneration);
        #expect(workerStartupConfiguration.configuredMaximumMlxMemoryBytes == 24_000_000_000);
        #expect(
            workerStartupConfiguration.globalPromptCacheRootDirectory
                == resolver.resolvedInstancePaths.promptCacheDirectory.string);
    }

    @Test
    func should_let_the_automatic_library_root_win_identity_collisions_with_authored_roots() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let stateDirectoryPath: String = temporaryRootPath + "/state";
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: stateDirectoryPath,
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[\"\(temporaryRootPath + "/authored-models")\"]}}");
        let instancePaths: AstronomicalInstancePaths = resolver.resolvedInstancePaths;
        try FileManager.default.createDirectory(atPath: instancePaths.modelsDirectory.string, withIntermediateDirectories: true);
        // The same identity lives in both the Library destination and an
        // authored root; Library publication owns the identity, and the
        // authored root's ambiguity cannot hide it.
        try self.writeModernbertModel(modelId: "CollisionBert", modelsRootPath: instancePaths.modelsDirectory.string);
        try self.writeModernbertModel(modelId: "CollisionBert", modelsRootPath: temporaryRootPath + "/authored-models");
        try self.writeModernbertModel(modelId: "AuthoredBert", modelsRootPath: temporaryRootPath + "/authored-models");

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();

        let discoveredModelIds: Set<String> = Set<String>(resolvedConfig.discoveredModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        });
        #expect(discoveredModelIds == Set<String>(["CollisionBert", "AuthoredBert"]));
        #expect(resolvedConfig.modelDiscoveryDiagnostics.isEmpty,
            "Library ownership clears the authored ambiguity for its identities: \(resolvedConfig.modelDiscoveryDiagnostics)");
        #expect(resolvedConfig.modelPolicyCatalog.count == 2);
    }

    @Test
    func should_treat_a_missing_automatic_models_directory_as_an_empty_library_without_creating_it() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: temporaryRootPath + "/state",
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]}}");
        let automaticModelsDirectory: FilePath = resolver.resolvedInstancePaths.modelsDirectory;

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();

        #expect(resolvedConfig.discoveredModels.isEmpty);
        #expect(resolvedConfig.configuredModelDirectories.isEmpty);
        #expect(FileManager.default.fileExists(atPath: automaticModelsDirectory.string) == false,
            "resolving must not materialize the automatic library root");
    }

    @Test
    func should_discover_the_automatic_organization_model_tree_before_configured_roots() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let stateDirectoryPath: String = temporaryRootPath + "/state";
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: stateDirectoryPath,
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[\"\(temporaryRootPath + "/configured-models")\"]}}");
        let automaticModelsDirectory: FilePath = resolver.resolvedInstancePaths.modelsDirectory;
        try FileManager.default.createDirectory(atPath: automaticModelsDirectory.string, withIntermediateDirectories: true);
        try self.writeModernbertModel(modelId: "AutomaticBert", modelsRootPath: automaticModelsDirectory.string);
        try self.writeModernbertModel(modelId: "ConfiguredBert", modelsRootPath: temporaryRootPath + "/configured-models");

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();

        let discoveredModelIds: Array<String> = resolvedConfig.discoveredModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        };
        #expect(discoveredModelIds == ["AutomaticBert", "ConfiguredBert"]);
        #expect(resolvedConfig.discoveredModels.allSatisfy({ (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
            return discoveredModel.modelSizeBytes > 0;
        }));
        #expect(resolvedConfig.configuredModelDirectories == [FilePath(string: temporaryRootPath + "/configured-models")]);
    }

    @Test
    func should_scan_a_repeated_automatic_root_once_while_preserving_authored_configuration() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        // Derive the automatic Library root from the instance paths the
        // resolver itself uses, so the authored config repeats the exact
        // root rather than a hardcoded layout assumption.
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryRootPath + "/state"),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let automaticModelsDirectory: FilePath = instancePaths.modelsDirectory;
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: temporaryRootPath + "/state",
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[\"\(automaticModelsDirectory.string)\"]}}");
        try FileManager.default.createDirectory(atPath: automaticModelsDirectory.string, withIntermediateDirectories: true);
        try self.writeModernbertModel(modelId: "RepeatedBert", modelsRootPath: automaticModelsDirectory.string);

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();

        #expect(resolvedConfig.discoveredModels.count == 1);
        #expect(resolvedConfig.configuredModelDirectories == [automaticModelsDirectory]);
    }

    @Test
    func should_start_when_the_automatic_models_path_is_not_a_directory() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: temporaryRootPath + "/state",
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]}}");
        let automaticModelsDirectory: FilePath = resolver.resolvedInstancePaths.modelsDirectory;
        try FileManager.default.createDirectory(
            atPath: (automaticModelsDirectory.string as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true);
        try Data("not a directory".utf8).write(to: URL(fileURLWithPath: automaticModelsDirectory.string));

        let resolvedConfig: ResolvedRuntimeConfig = try resolver.load();

        #expect(resolvedConfig.discoveredModels.isEmpty);
        #expect(resolvedConfig.modelDiscoveryDiagnostics.isEmpty);
    }

    @Test
    func should_return_a_typed_error_when_automatic_root_metadata_cannot_be_read() throws {
        let temporaryRootPath: String = try self.makeTemporaryRoot();
        let resolver: ResolvedRuntimeConfigResolver = try self.makeResolver(
            stateDirectoryPath: temporaryRootPath + "/state",
            configContents: "{\"$schema\":\"./astronomical-config.schema.json\",\"schema_version\":1,"
                + "\"runtime\":{\"model_directories\":[]}}");
        let automaticModelsDirectory: FilePath = resolver.resolvedInstancePaths.modelsDirectory;
        try FileManager.default.createDirectory(
            atPath: (automaticModelsDirectory.string as NSString).deletingLastPathComponent,
            withIntermediateDirectories: true);
        // A symlink loop makes every metadata read of the automatic root
        // fail: the resolver must surface that as a typed error instead of
        // silently serving an empty library.
        try FileManager.default.createSymbolicLink(
            atPath: automaticModelsDirectory.string,
            withDestinationPath: "models");

        let resolutionOutcome: Error? = {
            do {
                _ = try resolver.load();
                return nil;
            } catch let resolutionError {
                return resolutionError;
            }
        }();
        let resolutionError: DiscoveryDiscoveredModelError = try #require(
            resolutionOutcome as? DiscoveryDiscoveredModelError,
            "expected a typed discovery error, got \(String(describing: resolutionOutcome))");
        guard case let .readDirectory(directoryPath, _) = resolutionError else {
            Issue.record("expected an automatic-root metadata error, got \(resolutionError)");
            return;
        }
        #expect(directoryPath == automaticModelsDirectory);
    }
}
