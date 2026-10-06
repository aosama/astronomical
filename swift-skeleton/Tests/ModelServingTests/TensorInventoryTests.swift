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
}
