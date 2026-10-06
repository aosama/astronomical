import XCTest;
import ModelServing;

/// Behavioral journeys for the canonical tensor inventory, twin-porting the
/// pure-inventory tests of crates/model-serving/tests/hermetic/tensor_inventory.rs.
/// The partition-validator journeys in that Rust file depend on the bounded
/// safetensors source validator and land with that port.
final class TensorInventoryTests: XCTestCase {

    private func targetLocation(
        canonicalName: String, storedName: String, sourceId: TensorSourceId,
        declarationOrigin: TensorDeclarationOrigin) -> TensorLocation {
        return TensorLocation(
            canonicalName: canonicalName, storedName: storedName, sourceId: sourceId,
            semanticRole: .target, declarationOrigin: declarationOrigin);
    }

    func testShouldResolveCanonicalNamesToStoredSidecarNames() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 7);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.targetLocation(
            canonicalName: "language_model.trunk.fc.weight", storedName: "trunk.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        let location: TensorLocation = try XCTUnwrap(
            inventory.location(canonicalName: "language_model.trunk.fc.weight"),
            "the canonical tensor should resolve");
        XCTAssertEqual(location.storedName, "trunk.fc.weight");
        XCTAssertEqual(location.sourceId, sourceId);
    }

    func testShouldRejectEmbeddedAndSidecarCanonicalCollisions() throws {
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.targetLocation(
            canonicalName: "language_model.trunk.fc.weight",
            storedName: "language_model.trunk.fc.weight",
            sourceId: TensorSourceId(sourceNumber: 1), declarationOrigin: .mainIndex));

        do {
            try inventory.insert(location: self.targetLocation(
                canonicalName: "language_model.trunk.fc.weight", storedName: "trunk.fc.weight",
                sourceId: TensorSourceId(sourceNumber: 2),
                declarationOrigin: .architectureSidecar));
            XCTFail("the sidecar must not silently override the embedded location");
        } catch let collision as TensorInventoryError {
            XCTAssertEqual(
                collision, .canonicalNameCollision(canonicalName: "language_model.trunk.fc.weight"));
        }
    }

    func testShouldRejectDuplicatePhysicalTensorLocations() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 3);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.targetLocation(
            canonicalName: "language_model.trunk.fc.weight", storedName: "trunk.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        do {
            try inventory.insert(location: self.targetLocation(
                canonicalName: "language_model.trunk.alias.weight", storedName: "trunk.fc.weight",
                sourceId: sourceId, declarationOrigin: .architectureSidecar));
            XCTFail("one physical tensor must not have two canonical identities");
        } catch let duplicate as TensorInventoryError {
            XCTAssertEqual(
                duplicate, .physicalLocationCollision(
                    sourceId: sourceId, storedName: "trunk.fc.weight"));
        }
    }

    func testShouldRemoveRemovedCanonicalNamesFromEveryLookup() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 9);
        let inventory: TensorInventory = TensorInventory();
        var removedCanonicalNames: Set<String> = Set();
        for tensorSuffix: String in ["weight", "scales", "biases"] {
            let canonicalName: String = "language_model.trunk.proj.\(tensorSuffix)";
            try inventory.insert(location: self.targetLocation(
                canonicalName: canonicalName,
                storedName: "trunk.proj.\(tensorSuffix)",
                sourceId: sourceId, declarationOrigin: .architectureSidecar));
            removedCanonicalNames.insert(canonicalName);
        }

        inventory.removeCanonicalNames(canonicalNames: removedCanonicalNames);

        XCTAssertEqual(inventory.tensorCount(), 0);
        XCTAssertTrue(inventory.sourceIds().isEmpty);
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
