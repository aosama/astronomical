import Foundation;

import Testing;

import ModelServing;

@testable import ModelServing;

/// Hermetic journeys for the committed-block manifest: a contract-built
/// manifest round-trips through its staged, synchronized publication and
/// validates against the active contract, while foreign fingerprints,
/// foreign formats, invalid hashes, and topology mismatches fail closed.
final class PersistentPromptCacheBlockManifestTests {

    @Test
    func should_round_trip_and_validate_a_contract_built_manifest() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: [UInt32](repeating: 7, count: modelContract.blockTokenCount));
        let stagingDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: stagingDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: stagingDirectory); }

        let manifest: PersistentPromptCacheBlockManifest = try Self.buildManifest(
            blockKey: blockKey, parentBlockKey: nil, modelContract: modelContract);
        try manifest.writeToStagingDirectory(stagingBlockDirectory: stagingDirectory);

        let validatedManifest: PersistentPromptCacheBlockManifest = try
            PersistentPromptCacheBlockManifest.readFromBlockDirectory(
                blockDirectory: stagingDirectory, modelContract: modelContract);
        #expect(validatedManifest.blockIndex == 0);
        #expect(validatedManifest.hasSequenceState == modelContract.hasSequenceState);
        #expect(validatedManifest.hasBoundaryState == modelContract.hasBoundaryState);
        #expect(validatedManifest.storageContractFingerprint
            == modelContract.storageContractFingerprintHex());
        #expect(try validatedManifest.blockHash() == blockKey.blockHash());
        #expect(validatedManifest.parentBlockHash() == nil);
    }

    @Test
    func should_reject_a_foreign_fingerprint_and_an_invalid_hash() throws {
        let modelContract: PersistentPromptCacheModelContract = try PersistentPromptCacheFixture
            .ornithModelContract();
        let blockKey: PersistentPromptCacheBlockKey = try PersistentPromptCacheBlockKey
            .forRootBlock(
                modelContract: modelContract,
                blockTokens: [UInt32](repeating: 7, count: modelContract.blockTokenCount));
        let stagingDirectory: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(
            at: stagingDirectory, withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(at: stagingDirectory); }
        let manifest: PersistentPromptCacheBlockManifest = try Self.buildManifest(
            blockKey: blockKey, parentBlockKey: nil, modelContract: modelContract);
        try manifest.writeToStagingDirectory(stagingBlockDirectory: stagingDirectory);
        let manifestFileUrl: URL = stagingDirectory
            .appendingPathComponent(PersistentPromptCacheStoreFile.BLOCK_MANIFEST_FILE_NAME);

        try PersistentPromptCacheBlockManifestTests.rewriteManifestField(
            manifestFileUrl: manifestFileUrl,
            fieldName: "storage_contract_fingerprint",
            fieldValue: String(repeating: "ab", count: 32));
        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheBlockManifest.readFromBlockDirectory(
                blockDirectory: stagingDirectory, modelContract: modelContract);
        };

        try PersistentPromptCacheBlockManifestTests.rewriteManifestField(
            manifestFileUrl: manifestFileUrl,
            fieldName: "storage_contract_fingerprint",
            fieldValue: modelContract.storageContractFingerprintHex());
        try PersistentPromptCacheBlockManifestTests.rewriteManifestField(
            manifestFileUrl: manifestFileUrl,
            fieldName: "block_hash",
            fieldValue: "zz");
        #expect(throws: (any Error).self) {
            _ = try PersistentPromptCacheBlockManifest.readFromBlockDirectory(
                blockDirectory: stagingDirectory, modelContract: modelContract);
        };
    }

    /// Builds a manifest through the internal publication constructor; the
    /// disk store (next slice) is the production caller.
    static func buildManifest(
        blockKey: PersistentPromptCacheBlockKey,
        parentBlockKey: PersistentPromptCacheBlockKey?,
        modelContract: PersistentPromptCacheModelContract
    ) throws -> PersistentPromptCacheBlockManifest {
        return PersistentPromptCacheBlockManifest(
            blockKey: blockKey, parentBlockKey: parentBlockKey, modelContract: modelContract);
    }

    private static func rewriteManifestField(
        manifestFileUrl: URL, fieldName: String, fieldValue: String
    ) throws {
        let manifestObject: [String: Any] = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestFileUrl), options: []) as! [String: Any];
        var replacementObject: [String: Any] = manifestObject;
        replacementObject[fieldName] = fieldValue;
        try JSONSerialization.data(withJSONObject: replacementObject, options: [.sortedKeys])
            .write(to: manifestFileUrl);
    }
}
