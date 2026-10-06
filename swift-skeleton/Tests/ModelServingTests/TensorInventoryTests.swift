import XCTest;
import ModelServing;

/// Behavioral journeys for the canonical tensor inventory, twin-porting the
/// pure-inventory tests of crates/model-serving/tests/hermetic/tensor_inventory.rs.
/// The partition-validator journeys in that Rust file depend on the bounded
/// safetensors source validator and land with that port.
final class TensorInventoryTests: XCTestCase {

    private func mtpLocation(
        canonicalName: String, storedName: String, sourceId: TensorSourceId,
        declarationOrigin: TensorDeclarationOrigin) -> TensorLocation {
        return TensorLocation(
            canonicalName: canonicalName, storedName: storedName, sourceId: sourceId,
            semanticRole: .multiTokenPrediction, declarationOrigin: declarationOrigin,
            feature: .multiTokenPrediction);
    }

    func testShouldResolveCanonicalMtpNamesToStoredSidecarNames() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 7);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.mtpLocation(
            canonicalName: "language_model.mtp.fc.weight", storedName: "mtp.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        let location: TensorLocation = try XCTUnwrap(
            inventory.location(canonicalName: "language_model.mtp.fc.weight"),
            "the canonical MTP tensor should resolve");
        XCTAssertEqual(location.storedName, "mtp.fc.weight");
        XCTAssertEqual(location.sourceId, sourceId);
    }

    func testShouldRejectEmbeddedAndSidecarCanonicalCollisions() throws {
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.mtpLocation(
            canonicalName: "language_model.mtp.fc.weight",
            storedName: "language_model.mtp.fc.weight",
            sourceId: TensorSourceId(sourceNumber: 1), declarationOrigin: .mainIndex));

        do {
            try inventory.insert(location: self.mtpLocation(
                canonicalName: "language_model.mtp.fc.weight", storedName: "mtp.fc.weight",
                sourceId: TensorSourceId(sourceNumber: 2),
                declarationOrigin: .architectureSidecar));
            XCTFail("the sidecar must not silently override embedded MTP");
        } catch let collision as TensorInventoryError {
            XCTAssertEqual(
                collision, .canonicalNameCollision(canonicalName: "language_model.mtp.fc.weight"));
        }
    }

    func testShouldRejectDuplicatePhysicalTensorLocations() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 3);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.mtpLocation(
            canonicalName: "language_model.mtp.fc.weight", storedName: "mtp.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        do {
            try inventory.insert(location: self.mtpLocation(
                canonicalName: "language_model.mtp.alias.weight", storedName: "mtp.fc.weight",
                sourceId: sourceId, declarationOrigin: .architectureSidecar));
            XCTFail("one physical tensor must not have two canonical identities");
        } catch let duplicate as TensorInventoryError {
            XCTAssertEqual(
                duplicate, .physicalLocationCollision(
                    sourceId: sourceId, storedName: "mtp.fc.weight"));
        }
    }

    func testShouldRemoveTheCompleteOptionalMtpFeatureAfterACollision() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 9);
        let inventory: TensorInventory = TensorInventory();
        for tensorSuffix: String in ["weight", "scales", "biases"] {
            try inventory.insert(location: self.mtpLocation(
                canonicalName: "language_model.mtp.proj.\(tensorSuffix)",
                storedName: "mtp.proj.\(tensorSuffix)",
                sourceId: sourceId, declarationOrigin: .architectureSidecar));
        }

        inventory.removeFeature(feature: .multiTokenPrediction);

        XCTAssertEqual(inventory.tensorCount(), 0);
        XCTAssertTrue(inventory.sourceIds().isEmpty);
    }

    // MARK: - Partition-validator journeys (twin-port the three Rust
    // tensor_inventory tests that exercise the retained source validator).

    private func sharedTargetAndMtpFixture(mtpDtype: String) throws -> URL {
        let modelDirectory: URL = try Self.makeInventoryTemporaryDirectory();
        let header: String = "{\"language_model.target.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},"
            + "\"language_model.mtp.proj.weight\":{\"dtype\":\"\(mtpDtype)\",\"shape\":[1],\"data_offsets\":[2,4]}}";
        try Data(Self.framedBytes(
            headerText: header, payloadByteCount: 4)).write(
            to: modelDirectory.appendingPathComponent("model.safetensors"));
        return modelDirectory;
    }

    private func sharedTargetAndMtpProfiles() -> Array<TensorProfile> {
        return [
            TensorProfile(
                name: "language_model.target.weight", dtype: .bfloat16, shape: [1],
                equivalentPublishedShapes: []),
            TensorProfile(
                name: "language_model.mtp.proj.weight", dtype: .uint32, shape: [1],
                equivalentPublishedShapes: []),
        ];
    }

    private func sharedSourceInventory() throws -> TensorInventory {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 1);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: TensorLocation(
            canonicalName: "language_model.target.weight",
            storedName: "language_model.target.weight", sourceId: sourceId,
            semanticRole: .target, declarationOrigin: .mainIndex, feature: nil));
        try inventory.insert(location: self.mtpLocation(
            canonicalName: "language_model.mtp.proj.weight",
            storedName: "language_model.mtp.proj.weight", sourceId: sourceId,
            declarationOrigin: .mainIndex));
        return inventory;
    }

    func testShouldPreserveRequiredTargetProfilesWhenEmbeddedOptionalMtpHasTheWrongDtype() throws {
        // The physical source is structurally valid and its target tensor
        // matches. Only the optional MTP dtype conflicts with its canonical
        // profile, so target serving must remain available.
        let modelDirectory: URL = try self.sharedTargetAndMtpFixture(mtpDtype: "BF16");
        defer { try? FileManager.default.removeItem(at: modelDirectory); }

        let optionalMtpProfilesAreValid: Bool = try ValidatedSafetensorsSource
            .validateSafetensorsProfilePartitionsForTests(
                modelDirectory: modelDirectory.path, relativeFileName: "model.safetensors",
                inventory: try self.sharedSourceInventory(),
                canonicalProfiles: self.sharedTargetAndMtpProfiles(),
                optionalFeature: .multiTokenPrediction);

        XCTAssertFalse(optionalMtpProfilesAreValid);
    }

    func testShouldPreserveRequiredTargetWhenOptionalMtpUsesAKnownUnsupportedDtype() throws {
        // U16 is structurally valid SafeTensors storage but unsupported by this
        // MTP execution profile. Structural parsing must succeed so the
        // optional feature can be disabled without rejecting the valid target
        // tensor that shares this physical source.
        let modelDirectory: URL = try self.sharedTargetAndMtpFixture(mtpDtype: "U16");
        defer { try? FileManager.default.removeItem(at: modelDirectory); }

        let optionalMtpProfilesAreValid: Bool = try ValidatedSafetensorsSource
            .validateSafetensorsProfilePartitionsForTests(
                modelDirectory: modelDirectory.path, relativeFileName: "model.safetensors",
                inventory: try self.sharedSourceInventory(),
                canonicalProfiles: self.sharedTargetAndMtpProfiles(),
                optionalFeature: .multiTokenPrediction);

        XCTAssertFalse(optionalMtpProfilesAreValid);
    }

    func testShouldAcceptADeclaredPublishedShapeThroughTheRetainedSourceValidator() throws {
        let modelDirectory: URL = try Self.makeInventoryTemporaryDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectory); }
        let header: String = "{\"vision_tower.patch_embed.proj.weight\":{\"dtype\":\"F32\",\"shape\":[2,3,2,1,1],\"data_offsets\":[0,48]}}";
        try Data(Self.framedBytes(
            headerText: header, payloadByteCount: 48)).write(
            to: modelDirectory.appendingPathComponent("vision.safetensors"));
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 1);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: TensorLocation(
            canonicalName: "vision_tower.patch_embed.proj.weight",
            storedName: "vision_tower.patch_embed.proj.weight", sourceId: sourceId,
            semanticRole: .vision, declarationOrigin: .mainIndex, feature: nil));

        let declaredPublishedShapeAccepted: Bool = try ValidatedSafetensorsSource
            .validateSafetensorsProfilePartitionsForTests(
                modelDirectory: modelDirectory.path, relativeFileName: "vision.safetensors",
                inventory: inventory,
                canonicalProfiles: [TensorProfile(
                    name: "vision_tower.patch_embed.proj.weight", dtype: .float32,
                    shape: [2, 2, 1, 1, 3],
                    equivalentPublishedShapes: [[2, 3, 2, 1, 1]])],
                optionalFeature: .multiTokenPrediction);

        XCTAssertTrue(declaredPublishedShapeAccepted);
    }

    /// Little-endian length prefix plus header text plus zeroed payload,
    /// using the same immutable scalar-view pattern as SafetensorsHeaderTests.
    private static func framedBytes(headerText: String, payloadByteCount: Int) -> Array<UInt8> {
        var framedFileBytes: Array<UInt8> = Array();
        var littleEndianHeaderLength: UInt64 = UInt64(headerText.utf8.count);
        withUnsafeBytes(of: &littleEndianHeaderLength) { (valueBuffer: UnsafeRawBufferPointer) -> Void in
            framedFileBytes.append(contentsOf: Array(valueBuffer));
        };
        framedFileBytes.append(contentsOf: Array(headerText.utf8));
        framedFileBytes.append(contentsOf: Array<UInt8>(repeating: 0, count: payloadByteCount));
        return framedFileBytes;
    }

    private static func makeInventoryTemporaryDirectory() throws -> URL {
        let directoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tensor-inventory-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: directoryUrl, withIntermediateDirectories: true);
        return directoryUrl;
    }
}
