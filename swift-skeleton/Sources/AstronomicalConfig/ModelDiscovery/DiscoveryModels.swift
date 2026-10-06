import Foundation;

import CryptoKit;

/// Neutral scan orchestration over configured model roots: the recursive
/// walker, per-family executable projection, and the duplicate/ambiguous
/// identity rejections.
///
/// Migrates crates/config/src/model_discovery/{mod.rs,artifact_discovery.rs,
/// classified_artifacts.rs provenance helpers}. Walks up to 4 levels deep so
/// Hub-style layouts (`hub/models--org--repo/snapshots/<hash>/`) and flat
/// libraries (`models/Org-Model-OptiQ-4bit/`) both discover.
public enum DiscoveryModels {

    private static let MAXIMUM_SCAN_DEPTH: Int = 4;
    private static let MAXIMUM_PUBLIC_DISCOVERY_DIAGNOSTICS: Int = 32;
    private static let LIBRARY_PROVENANCE_FILE_NAME: String = ".astronomical-library-provenance.json";
    private static let MAXIMUM_LIBRARY_PROVENANCE_BYTES: UInt64 = 16_384;

    // MARK: - Public entry points

    /** Per-root scans with duplicate identities rejected outright. */
    public static func discoverModels(
        modelDirectories: Array<FilePath>
    ) throws -> Array<DiscoveryModelDiscoveryDirectoryScan> {
        var directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan> = Array<DiscoveryModelDiscoveryDirectoryScan>();
        for modelDirectory: FilePath in modelDirectories {
            let discoveredModels: Array<DiscoveryDiscoveredModel>;
            switch (try DiscoveryModels.scanDirectoryForExecutableModels(rootDirectory: modelDirectory)) {
            case .available(let scannedModels): discoveredModels = scannedModels;
            case .unavailable: discoveredModels = Array<DiscoveryDiscoveredModel>();
            }
            directoryScans.append(DiscoveryModelDiscoveryDirectoryScan(
                path: modelDirectory,
                discoveredModels: discoveredModels));
        }
        try DiscoveryModels.rejectDuplicateModelIds(directoryScans: &directoryScans);
        return directoryScans;
    }

    /**
     * Preserves available models while excluding identities found beneath
     * multiple authored roots, surfacing bounded diagnostics instead.
     */
    public static func discoverModelsExcludingAmbiguousIdentities(
        modelDirectories: Array<FilePath>
    ) throws -> DiscoveryModelDiscoveryReport {
        var directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan> = Array<DiscoveryModelDiscoveryDirectoryScan>();
        var scannedCanonicalDirectories: Set<String> = Set<String>();
        var diagnostics: Array<DiscoveryModelDiscoveryDiagnostic> = Array<DiscoveryModelDiscoveryDiagnostic>();
        for (rootIndex, modelDirectory) in modelDirectories.enumerated() {
            var discoveredModels: Array<DiscoveryDiscoveredModel> = Array<DiscoveryDiscoveredModel>();
            if scannedCanonicalDirectories.insert(modelDirectory.string).inserted {
                switch (try DiscoveryModels.scanDirectoryForExecutableModels(rootDirectory: modelDirectory)) {
                case .available(let scannedModels): discoveredModels = scannedModels;
                case .unavailable:
                    if diagnostics.count < DiscoveryModels.MAXIMUM_PUBLIC_DISCOVERY_DIAGNOSTICS {
                        diagnostics.append(DiscoveryModelDiscoveryDiagnostic.unavailableModelDirectory(
                            configuredRootNumber: rootIndex + 1));
                    }
                }
            }
            directoryScans.append(DiscoveryModelDiscoveryDirectoryScan(
                path: modelDirectory,
                discoveredModels: discoveredModels));
        }

        var modelIdToRootNumbers: Dictionary<String, Set<Int>> = Dictionary<String, Set<Int>>();
        for (rootIndex, directoryScan) in directoryScans.enumerated() {
            for discoveredModel: DiscoveryDiscoveredModel in directoryScan.discoveredModels {
                modelIdToRootNumbers[discoveredModel.modelId, default: Set<Int>()].insert(rootIndex + 1);
            }
        }
        var ambiguousModelIds: Set<String> = Set<String>();
        for (modelId, configuredRootNumbers) in modelIdToRootNumbers {
            guard configuredRootNumbers.count >= 2 else {
                continue;
            }
            ambiguousModelIds.insert(modelId);
            if diagnostics.count < DiscoveryModels.MAXIMUM_PUBLIC_DISCOVERY_DIAGNOSTICS {
                diagnostics.append(DiscoveryModelDiscoveryDiagnostic.ambiguousModelIdentity(
                    modelId: modelId,
                    configuredRootNumbers: configuredRootNumbers.sorted()));
            }
        }
        for scanIndex: Int in directoryScans.indices {
            directoryScans[scanIndex].discoveredModels.removeAll { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
                return ambiguousModelIds.contains(discoveredModel.modelId);
            };
        }
        return DiscoveryModelDiscoveryReport(
            directoryScans: directoryScans,
            diagnostics: diagnostics);
    }

    // MARK: - Recursive walker

    private enum ScannedModelRoot {
        case available(Array<DiscoveryDiscoveredModel>);
        case unavailable;
    }

    private static func scanDirectoryForExecutableModels(
        rootDirectory: FilePath
    ) throws -> ScannedModelRoot {
        var discoveredModels: Array<DiscoveryDiscoveredModel> = Array<DiscoveryDiscoveredModel>();
        do {
            try DiscoveryModels.scanDirectoryRecursive(
                currentDirectory: rootDirectory,
                depth: 0,
                discoveredModels: &discoveredModels);
        } catch {
            // An unreadable root is "unavailable", not fatal; unreadable
            // subdirectories are skipped silently by the recursion.
            return .unavailable;
        }
        return .available(discoveredModels);
    }

    private static func scanDirectoryRecursive(
        currentDirectory: FilePath,
        depth: Int,
        discoveredModels: inout Array<DiscoveryDiscoveredModel>
    ) throws -> Void {
        if depth > DiscoveryModels.MAXIMUM_SCAN_DEPTH {
            return;
        }
        // A HuggingFace cache entry (models--org--repo/) resolves to its
        // active snapshot and discovers from there with the decoded id.
        if let directoryName: String = DiscoveryPathNavigation.lastComponentName(of: currentDirectory),
           directoryName.hasPrefix("models--"),
           let cacheSnapshot: DiscoveryHuggingFaceCache.Snapshot = DiscoveryHuggingFaceCache.resolveCacheEntry(
            huggingFaceCacheDirectory: currentDirectory) {
            let leafModelId: String = DiscoveryHuggingFaceCache.leafModelId(ofDecodedModelId: cacheSnapshot.modelId);
            if let discoveredModel: DiscoveryDiscoveredModel = DiscoveryModels.tryDiscoverModel(
                modelDirectory: cacheSnapshot.snapshotDirectory,
                modelId: leafModelId) {
                discoveredModels.append(discoveredModel);
            }
            return;
        }

        let hasPipelineIndex: Bool = DiscoveryPathNavigation.isExistingRegularFile(
            path: currentDirectory.appending(component: "model_index.json"));
        let hasConfigJson: Bool = DiscoveryPathNavigation.isExistingRegularFile(
            path: currentDirectory.appending(component: "config.json"));
        if (hasConfigJson || hasPipelineIndex),
           let discoveredModel: DiscoveryDiscoveredModel = DiscoveryModels.tryDiscoverModel(
            modelDirectory: currentDirectory,
            modelId: DiscoveryPathNavigation.lastComponentName(of: currentDirectory) ?? "unknown") {
            discoveredModels.append(discoveredModel);
            // Don't recurse into a model directory — it won't contain
            // nested models.
            return;
        }
        // A Diffusers pipeline root is terminal even when unsupported or
        // incomplete; nested component configs are not independently
        // requestable artifacts.
        if hasPipelineIndex {
            return;
        }

        let directoryEntryNames: Array<String>;
        do {
            directoryEntryNames = try FileManager.default.contentsOfDirectory(atPath: currentDirectory.string);
        } catch {
            // Silently skip directories we can't read (permissions, etc.);
            // only an unreadable root becomes an unavailable scan.
            if depth == 0 {
                throw error;
            }
            return;
        }
        for directoryEntryName: String in directoryEntryNames {
            guard !directoryEntryName.hasPrefix(".") else {
                continue;
            }
            let entryPath: FilePath = currentDirectory.appending(component: directoryEntryName);
            var isDirectory: ObjCBool = ObjCBool(false);
            guard FileManager.default.fileExists(atPath: entryPath.string, isDirectory: &isDirectory), isDirectory.boolValue else {
                continue;
            }
            try? DiscoveryModels.scanDirectoryRecursive(
                currentDirectory: entryPath,
                depth: depth + 1,
                discoveredModels: &discoveredModels);
        }
    }

    // MARK: - Per-family executable projection

    /** Discovers one executable model from a classified family root. */
    public static func tryDiscoverModel(
        modelDirectory: FilePath,
        modelId: String
    ) -> DiscoveryDiscoveredModel? {
        // The typed classifier rejects ambiguous duplicate family markers
        // before the looser metadata document can participate in discovery;
        // a classification failure skips the directory like the Rust
        // `ok().flatten()`.
        let classifiedFamily: ModelFamily?? = try? FamilyDiscovery.classifyModelDirectory(
            modelDirectory: modelDirectory,
            attributionEnabled: false);
        guard let modelFamily: ModelFamily = classifiedFamily ?? nil else {
            return nil;
        }
        switch (modelFamily) {
        case .qwen35:
            return DiscoveryModels.discoverChatBackedModel(
                modelDirectory: modelDirectory,
                modelId: modelId,
                modelFamily: modelFamily,
                license: nil) { (configObject: Dictionary<String, Any>) -> DiscoveryChatModelCapabilities? in
                guard let metadata: Qwen35.DiscoveredModelMetadata = Qwen35.discoverModelMetadata(
                    modelDirectory: modelDirectory,
                    configObject: configObject) else {
                    return nil;
                }
                return DiscoveryChatModelCapabilities(
                    contextWindowTokens: metadata.contextWindowTokens,
                    maximumInputTokens: metadata.maximumInputTokens,
                    maximumOutputTokens: metadata.maximumOutputTokens,
                    supportsVision: metadata.hasVision,
                    supportsReasoning: metadata.supportsReasoning,
                    supportsToolCalls: metadata.supportsToolCalls);
            };
        case .k2HorizonMova:
            guard let configBytes: Data = DiscoveryModels.readFileBytes(
                path: modelDirectory.appending(component: "config.json")) else {
                return nil;
            }
            guard let metadata: K2HorizonMova.DiscoveredModelMetadata = K2HorizonMova.discoverModelMetadata(
                modelDirectory: modelDirectory,
                configBytes: configBytes) else {
                return nil;
            }
            let provenance: (providerModelId: String, revision: String)? = DiscoveryModels.immutableModelProvenance(
                modelDirectory: modelDirectory);
            return DiscoveryDiscoveredModel(
                modelId: modelId,
                providerModelId: provenance?.providerModelId,
                modelFamily: modelFamily,
                revision: provenance?.revision
                    ?? DiscoveryModels.deriveRevisionFromConfigBytes(configBytes: configBytes),
                modelDirectory: modelDirectory,
                capabilities: .chat(DiscoveryChatModelCapabilities(
                    contextWindowTokens: metadata.contextWindowTokens,
                    maximumInputTokens: metadata.maximumInputTokens,
                    maximumOutputTokens: metadata.maximumOutputTokens,
                    supportsVision: metadata.hasVision,
                    supportsReasoning: metadata.supportsReasoning,
                    supportsToolCalls: metadata.supportsToolCalls)),
                license: ModelLicense.apache20,
                modelSizeBytes: metadata.modelSizeBytes);
        case .modernbert:
            return DiscoveryModels.discoverEmbeddingsModel(
                modelDirectory: modelDirectory,
                modelId: modelId) { (configObject: Dictionary<String, Any>) -> DiscoveryEmbeddingModelCapabilities? in
                guard let metadata: Modernbert.DiscoveredModelMetadata = Modernbert.discoverModelMetadata(
                    modelDirectory: modelDirectory,
                    configObject: configObject) else {
                    return nil;
                }
                return DiscoveryEmbeddingModelCapabilities(
                    vectorWidth: metadata.vectorWidth,
                    maximumInputTokens: metadata.maximumInputTokens);
            };
        case .flux2Klein:
            guard let evidence: Flux2Klein.DirectoryEvidence = try? Flux2Klein.verifyModelDirectory(
                modelDirectory: modelDirectory) else {
                return nil;
            }
            return DiscoveryDiscoveredModel(
                modelId: evidence.canonicalModelId,
                providerModelId: evidence.providerModelId,
                modelFamily: modelFamily,
                revision: evidence.revision,
                modelDirectory: modelDirectory,
                capabilities: .imageGeneration(evidence.capabilities),
                license: evidence.license,
                modelSizeBytes: evidence.modelSizeBytes);
        case .qwenImage21:
            guard let evidence: QwenImage21.DirectoryEvidence = try? QwenImage21.verifyModelDirectory(
                modelDirectory: modelDirectory) else {
                return nil;
            }
            return DiscoveryDiscoveredModel(
                modelId: evidence.canonicalModelId,
                providerModelId: evidence.providerModelId,
                modelFamily: modelFamily,
                revision: evidence.revision,
                modelDirectory: modelDirectory,
                capabilities: .imageGeneration(evidence.capabilities),
                license: evidence.license,
                modelSizeBytes: evidence.modelSizeBytes);
        }
    }

    /** Projects a chat-capability family that reads a JSON config object. */
    private static func discoverChatBackedModel(
        modelDirectory: FilePath,
        modelId: String,
        modelFamily: ModelFamily,
        license: ModelLicense?,
        projectCapabilities: (Dictionary<String, Any>) -> DiscoveryChatModelCapabilities?
    ) -> DiscoveryDiscoveredModel? {
        guard let configBytes: Data = DiscoveryModels.readFileBytes(
            path: modelDirectory.appending(component: "config.json")) else {
            return nil;
        }
        guard let configRootValue: Any = try? DiscoveryStrictJsonDocument.parseDocument(bytes: configBytes),
              let configObject: Dictionary<String, Any> = configRootValue as? Dictionary<String, Any> else {
            return nil;
        }
        guard let capabilities: DiscoveryChatModelCapabilities = projectCapabilities(configObject) else {
            return nil;
        }
        let provenance: (providerModelId: String, revision: String)? = DiscoveryModels.immutableModelProvenance(
            modelDirectory: modelDirectory);
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: provenance?.providerModelId,
            modelFamily: modelFamily,
            revision: provenance?.revision
                ?? DiscoveryModels.deriveRevisionFromConfigBytes(configBytes: configBytes),
            modelDirectory: modelDirectory,
            capabilities: .chat(capabilities),
            license: license,
            modelSizeBytes: 0);
    }

    /** Projects the embeddings family, which reads a JSON config object. */
    private static func discoverEmbeddingsModel(
        modelDirectory: FilePath,
        modelId: String,
        projectCapabilities: (Dictionary<String, Any>) -> DiscoveryEmbeddingModelCapabilities?
    ) -> DiscoveryDiscoveredModel? {
        guard let configBytes: Data = DiscoveryModels.readFileBytes(
            path: modelDirectory.appending(component: "config.json")) else {
            return nil;
        }
        guard let configRootValue: Any = try? DiscoveryStrictJsonDocument.parseDocument(bytes: configBytes),
              let configObject: Dictionary<String, Any> = configRootValue as? Dictionary<String, Any> else {
            return nil;
        }
        guard let capabilities: DiscoveryEmbeddingModelCapabilities = projectCapabilities(configObject) else {
            return nil;
        }
        let provenance: (providerModelId: String, revision: String)? = DiscoveryModels.immutableModelProvenance(
            modelDirectory: modelDirectory);
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: provenance?.providerModelId,
            modelFamily: .modernbert,
            revision: provenance?.revision
                ?? DiscoveryModels.deriveRevisionFromConfigBytes(configBytes: configBytes),
            modelDirectory: modelDirectory,
            capabilities: .embeddings(capabilities),
            license: nil,
            modelSizeBytes: 0);
    }

    // MARK: - Identity rejections

    private static func rejectDuplicateModelIds(
        directoryScans: inout Array<DiscoveryModelDiscoveryDirectoryScan>
    ) throws -> Void {
        var seenModelIds: Set<String> = Set<String>();
        for directoryScan: DiscoveryModelDiscoveryDirectoryScan in directoryScans {
            for discoveredModel: DiscoveryDiscoveredModel in directoryScan.discoveredModels {
                guard seenModelIds.insert(discoveredModel.modelId).inserted else {
                    throw DiscoveryDiscoveredModelError.duplicateModelId(
                        modelId: discoveredModel.modelId,
                        modelDirectories: [directoryScan.path]);
                }
            }
        }
    }

    // MARK: - Immutable provenance and revision derivation

    /**
     * Reads the library provenance file written at download time; nil when
     * absent, oversized, or not schema-conformant.
     */
    public static func immutableModelProvenance(
        modelDirectory: FilePath
    ) -> (providerModelId: String, revision: String)? {
        let provenanceFilePath: FilePath = modelDirectory.appending(component: DiscoveryModels.LIBRARY_PROVENANCE_FILE_NAME);
        guard let provenanceAttributes: [FileAttributeKey: Any] = try? FileManager.default.attributesOfItem(
            atPath: provenanceFilePath.string) else {
            return nil;
        }
        let fileSizeBytes: UInt64 = (provenanceAttributes[.size] as? NSNumber)?.uint64Value ?? 0;
        guard fileSizeBytes > 0 && fileSizeBytes <= DiscoveryModels.MAXIMUM_LIBRARY_PROVENANCE_BYTES else {
            return nil;
        }
        guard let provenanceBytes: Data = DiscoveryModels.readFileBytes(path: provenanceFilePath),
              let provenanceObject: Any = try? DiscoveryStrictJsonDocument.parseDocument(bytes: provenanceBytes),
              let provenanceFields: Dictionary<String, Any> = provenanceObject as? Dictionary<String, Any> else {
            return nil;
        }
        guard let schemaVersion: UInt64 = (provenanceFields["schema_version"] as? NSNumber)?.uint64Value,
              schemaVersion == 1,
              let providerModelId: String = provenanceFields["provider_model_id"] as? String,
              let revision: String = provenanceFields["revision"] as? String else {
            return nil;
        }
        let revisionCharacters: Set<Character> = Set<Character>("0123456789abcdef");
        guard revision.count == 40 && revision.allSatisfy({ (revisionCharacter: Character) -> Bool in
            return revisionCharacters.contains(revisionCharacter);
        }) else {
            return nil;
        }
        let identityComponents: Array<String> = providerModelId.split(separator: "/").map { (identityComponent: Substring) -> String in
            return String(identityComponent);
        };
        guard identityComponents.count == 2,
              !identityComponents[0].isEmpty, !identityComponents[1].isEmpty else {
            return nil;
        }
        return (providerModelId: providerModelId, revision: revision);
    }

    /// The config digest stands in for a revision when no immutable
    /// provenance exists: the first 12 hex characters of the SHA-256, the
    /// big-endian top 6 bytes of the hash.
    public static func deriveRevisionFromConfigBytes(configBytes: Data) -> String {
        let configDigest: SHA256Digest = SHA256.hash(data: configBytes);
        let digestBytes: Array<UInt8> = configDigest.map { (digestByte: UInt8) -> UInt8 in
            return digestByte;
        };
        return digestBytes.prefix(6).map { (digestByte: UInt8) -> String in
            return String(format: "%02x", digestByte);
        }.joined();
    }

    private static func readFileBytes(path: FilePath) -> Data? {
        return FileManager.default.contents(atPath: path.string);
    }
}
