import Foundation;
import XCTest;

@testable import AstronomicalConfig;

/**
 * Hermetic journeys for the neutral discovery entry points: the recursive
 * walker, per-family executable projection, duplicate and ambiguous identity
 * handling, and unavailable-root diagnostics. Every fixture is synthesized
 * in a temporary directory — no downloads, no real artifacts.
 */
final class DiscoveryModelsTests: XCTestCase {

    private var temporaryRootPath: String?;

    override func tearDown() {
        if let temporaryRootPath: String = self.temporaryRootPath {
            try? FileManager.default.removeItem(atPath: temporaryRootPath);
            self.temporaryRootPath = nil;
        }
        super.tearDown();
    }

    private func makeTemporaryRoot() throws -> String {
        let temporaryRootPath: String = NSTemporaryDirectory() + "adisc-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryRootPath, withIntermediateDirectories: true);
        self.temporaryRootPath = temporaryRootPath;
        return temporaryRootPath;
    }

    private func writeFile(relativePath: String, contents: String, beneathRoot rootPath: String) throws -> FilePath {
        let absolutePath: String = rootPath + "/" + relativePath;
        let directoryPath: String = (absolutePath as NSString).deletingLastPathComponent;
        try FileManager.default.createDirectory(atPath: directoryPath, withIntermediateDirectories: true);
        try contents.write(to: URL(fileURLWithPath: absolutePath), atomically: true, encoding: String.Encoding.utf8);
        return FilePath(string: absolutePath);
    }

    /** A minimal executable ModernBERT directory: config, weights, tokenizer. */
    @discardableResult
    private func writeModernbertModel(relativePath: String, beneathRoot rootPath: String) throws -> FilePath {
        let configFilePath: FilePath = try self.writeFile(
            relativePath: relativePath + "/config.json",
            contents: "{\"model_type\":\"modernbert\",\"hidden_size\":768,"
                + "\"max_position_embeddings\":8192,\"quantization\":{\"bits\":8}}",
            beneathRoot: rootPath);
        // Completeness requires resident weights and a tokenizer alongside
        // the classified config; the reviewed profile is affine 8-bit and
        // the weight file must be measurable.
        try Data([0x01, 0x02, 0x03]).write(to: URL(fileURLWithPath: rootPath + "/" + relativePath + "/model.safetensors"));
        try "{}".write(
            to: URL(fileURLWithPath: rootPath + "/" + relativePath + "/tokenizer.json"),
            atomically: true,
            encoding: String.Encoding.utf8);
        return configFilePath;
    }

    private func qwen35ConfigJson() -> String {
        return "{\"model_type\":\"qwen3_5\",\"max_position_embeddings\":40960,"
            + "\"text_config\":{\"num_hidden_layers\":4,\"hidden_size\":1024}}";
    }

    func testDiscoverFindsAModernbertModelBeneathANestedRoot() throws {
        let rootPath: String = try self.makeTemporaryRoot();
        try self.writeModernbertModel(relativePath: "models/ModernBert-Retrieval", beneathRoot: rootPath);

        let directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan> = try DiscoveryModels.discoverModels(
            modelDirectories: [FilePath(string: rootPath)]);

        XCTAssertEqual(directoryScans.count, 1);
        XCTAssertEqual(directoryScans[0].discoveredModels.count, 1);
        let discoveredModel: DiscoveryDiscoveredModel = directoryScans[0].discoveredModels[0];
        XCTAssertEqual(discoveredModel.modelId, "ModernBert-Retrieval");
        XCTAssertEqual(discoveredModel.modelFamily, ModelFamily.modernbert);
        guard case let .embeddings(embeddingCapabilities) = discoveredModel.capabilities else {
            return XCTFail("expected embeddings capabilities, got \(discoveredModel.capabilities)");
        }
        XCTAssertEqual(embeddingCapabilities.vectorWidth, 768);
        XCTAssertEqual(embeddingCapabilities.maximumInputTokens, 8192);
    }

    func testWalkerStopsAtAPipelineRootAndSkipsHiddenDirectories() throws {
        let rootPath: String = try self.makeTemporaryRoot();
        // A Diffusers pipeline root is terminal: nested component configs
        // are not independently requestable.
        try self.writeFile(
            relativePath: "pipelines/flux-pipeline/model_index.json",
            contents: "{\"_class_name\":\"FluxPipeline\"}",
            beneathRoot: rootPath);
        try self.writeModernbertModel(
            relativePath: "pipelines/flux-pipeline/text_encoder",
            beneathRoot: rootPath);
        try self.writeModernbertModel(
            relativePath: ".hidden-cache/models/HiddenBert",
            beneathRoot: rootPath);

        let report: DiscoveryModelDiscoveryReport = try DiscoveryModels.discoverModelsExcludingAmbiguousIdentities(
            modelDirectories: [FilePath(string: rootPath)]);

        var discoveredModelIds: Set<String> = Set<String>();
        for directoryScan: DiscoveryModelDiscoveryDirectoryScan in report.directoryScans {
            for discoveredModel: DiscoveryDiscoveredModel in directoryScan.discoveredModels {
                discoveredModelIds.insert(discoveredModel.modelId);
            }
        }
        XCTAssertEqual(
            discoveredModelIds,
            Set<String>(),
            "pipeline roots and hidden directories must not yield executable models");
    }

    func testDuplicateIdentityAcrossRootsIsRejectedOutright() throws {
        let firstRootPath: String = try self.makeTemporaryRoot();
        let secondRootPath: String = NSTemporaryDirectory() + "adisc-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: secondRootPath, withIntermediateDirectories: true);
        defer {
            try? FileManager.default.removeItem(atPath: secondRootPath);
        }
        try self.writeModernbertModel(relativePath: "m/SharedBert", beneathRoot: firstRootPath);
        try self.writeModernbertModel(relativePath: "m/SharedBert", beneathRoot: secondRootPath);

        XCTAssertThrowsError(try DiscoveryModels.discoverModels(
            modelDirectories: [FilePath(string: firstRootPath), FilePath(string: secondRootPath)])) { (thrownError: any Error) in
            guard case DiscoveryDiscoveredModelError.duplicateModelId = thrownError else {
                return XCTFail("expected a duplicate identity rejection, got \(thrownError)");
            }
        }
    }

    func testAmbiguousIdentityIsDiagnosedAndExcludedInsteadOfRejected() throws {
        let firstRootPath: String = try self.makeTemporaryRoot();
        let secondRootPath: String = NSTemporaryDirectory() + "adisc-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: secondRootPath, withIntermediateDirectories: true);
        defer {
            try? FileManager.default.removeItem(atPath: secondRootPath);
        }
        try self.writeModernbertModel(relativePath: "m/SharedBert", beneathRoot: firstRootPath);
        try self.writeModernbertModel(relativePath: "m/SharedBert", beneathRoot: secondRootPath);
        try self.writeModernbertModel(relativePath: "m/UniqueBert", beneathRoot: firstRootPath);

        let report: DiscoveryModelDiscoveryReport = try DiscoveryModels.discoverModelsExcludingAmbiguousIdentities(
            modelDirectories: [FilePath(string: firstRootPath), FilePath(string: secondRootPath)]);

        var survivingModelIds: Set<String> = Set<String>();
        for directoryScan: DiscoveryModelDiscoveryDirectoryScan in report.directoryScans {
            for discoveredModel: DiscoveryDiscoveredModel in directoryScan.discoveredModels {
                survivingModelIds.insert(discoveredModel.modelId);
            }
        }
        XCTAssertEqual(survivingModelIds, Set<String>(["UniqueBert"]));
        XCTAssertTrue(report.diagnostics.contains { (diagnostic: DiscoveryModelDiscoveryDiagnostic) -> Bool in
            return diagnostic.code == DiscoveryModelDiscoveryDiagnosticCode.ambiguousModelIdentity
                && diagnostic.modelId == "SharedBert";
        }, "the ambiguous identity should be diagnosed: \(report.diagnostics)");
    }

    func testUnavailableRootIsDiagnosedAndTheRemainingRootsStillDiscover() throws {
        let availableRootPath: String = try self.makeTemporaryRoot();
        try self.writeModernbertModel(relativePath: "m/SurvivingBert", beneathRoot: availableRootPath);
        let missingRootPath: String = availableRootPath + "/does-not-exist";

        let report: DiscoveryModelDiscoveryReport = try DiscoveryModels.discoverModelsExcludingAmbiguousIdentities(
            modelDirectories: [FilePath(string: missingRootPath), FilePath(string: availableRootPath)]);

        XCTAssertTrue(report.diagnostics.contains { (diagnostic: DiscoveryModelDiscoveryDiagnostic) -> Bool in
            return diagnostic.code == DiscoveryModelDiscoveryDiagnosticCode.unavailableModelDirectory
                && diagnostic.configuredRootNumbers == [1];
        }, "the missing root should be diagnosed: \(report.diagnostics)");
        var survivingModelIds: Set<String> = Set<String>();
        for directoryScan: DiscoveryModelDiscoveryDirectoryScan in report.directoryScans {
            for discoveredModel: DiscoveryDiscoveredModel in directoryScan.discoveredModels {
                survivingModelIds.insert(discoveredModel.modelId);
            }
        }
        XCTAssertEqual(survivingModelIds, Set<String>(["SurvivingBert"]));
    }

    func testLibraryProvenanceProvidesImmutableIdentityAndRevision() throws {
        let rootPath: String = try self.makeTemporaryRoot();
        let configFilePath: FilePath = try self.writeModernbertModel(
            relativePath: "m/ProvenanceBert",
            beneathRoot: rootPath);
        // The written path points at config.json; provenance lives beside it.
        let modelDirectoryPath: String = (configFilePath.string as NSString).deletingLastPathComponent;
        let provenanceDirectory: FilePath = FilePath(string: modelDirectoryPath);
        let provenanceRevision: String = String(repeating: "a", count: 40);
        let provenanceJson: String = "{\"schema_version\":1,\"provider_model_id\":\"org/provenance-bert\","
            + "\"revision\":\"\(provenanceRevision)\"}";
        try provenanceJson.write(
            to: URL(fileURLWithPath: provenanceDirectory.appending(component: ".astronomical-library-provenance.json").string),
            atomically: true,
            encoding: String.Encoding.utf8);

        let provenance: (providerModelId: String, revision: String)? = DiscoveryModels.immutableModelProvenance(
            modelDirectory: provenanceDirectory);
        XCTAssertEqual(provenance?.providerModelId, "org/provenance-bert");
        XCTAssertEqual(provenance?.revision, String(repeating: "a", count: 40));

        let directoryScans: Array<DiscoveryModelDiscoveryDirectoryScan> = try DiscoveryModels.discoverModels(
            modelDirectories: [FilePath(string: rootPath)]);
        let discoveredModel: DiscoveryDiscoveredModel = directoryScans[0].discoveredModels[0];
        XCTAssertEqual(discoveredModel.providerModelId, "org/provenance-bert");
        XCTAssertEqual(discoveredModel.revision, String(repeating: "a", count: 40));
    }

    func testRevisionFallsBackToTheConfigDigestWithoutProvenance() throws {
        let configBytes: Data = Data("{\"model_type\":\"qwen3_5\"}".utf8);
        let derivedRevision: String = DiscoveryModels.deriveRevisionFromConfigBytes(configBytes: configBytes);
        XCTAssertEqual(derivedRevision.count, 12);
        let repeatedRevision: String = DiscoveryModels.deriveRevisionFromConfigBytes(configBytes: configBytes);
        XCTAssertEqual(derivedRevision, repeatedRevision);
    }
}
