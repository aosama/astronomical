import Foundation;
import ModelServing;
import Testing;
import JourneyCategories;

/**
 * Behavioral journeys for the canonical tensor inventory, twin-porting the
 * pure-inventory tests of crates/model-serving/tests/hermetic/tensor_inventory.rs.
 * The partition-validator journeys in that Rust file depend on the bounded
 * safetensors source validator and land with that port.
 */
@Suite(.tags(.hermeticJourney))
final class TensorInventoryTests {

    private func targetLocation(
        canonicalName: String, storedName: String, sourceId: TensorSourceId,
        declarationOrigin: TensorDeclarationOrigin) -> TensorLocation {
        return TensorLocation(
            canonicalName: canonicalName, storedName: storedName, sourceId: sourceId,
            semanticRole: .target, declarationOrigin: declarationOrigin);
    }

    @Test
    func should_resolve_canonical_names_to_stored_sidecar_names() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 7);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.targetLocation(
            canonicalName: "language_model.trunk.fc.weight", storedName: "trunk.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        let location: TensorLocation = try #require(
            inventory.location(canonicalName: "language_model.trunk.fc.weight"),
            "the canonical tensor should resolve");
        #expect(location.storedName == "trunk.fc.weight");
        #expect(location.sourceId == sourceId);
    }

    @Test
    func should_reject_embedded_and_sidecar_canonical_collisions() throws {
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
            Issue.record("the sidecar must not silently override the embedded location");
        } catch let collision as TensorInventoryError {
            #expect(
                collision == TensorInventoryError.canonicalNameCollision(
                    canonicalName: "language_model.trunk.fc.weight"));
        }
    }

    @Test
    func should_reject_duplicate_physical_tensor_locations() throws {
        let sourceId: TensorSourceId = TensorSourceId(sourceNumber: 3);
        let inventory: TensorInventory = TensorInventory();
        try inventory.insert(location: self.targetLocation(
            canonicalName: "language_model.trunk.fc.weight", storedName: "trunk.fc.weight",
            sourceId: sourceId, declarationOrigin: .architectureSidecar));

        do {
            try inventory.insert(location: self.targetLocation(
                canonicalName: "language_model.trunk.alias.weight", storedName: "trunk.fc.weight",
                sourceId: sourceId, declarationOrigin: .architectureSidecar));
            Issue.record("one physical tensor must not have two canonical identities");
        } catch let duplicate as TensorInventoryError {
            #expect(
                duplicate == TensorInventoryError.physicalLocationCollision(
                    sourceId: sourceId, storedName: "trunk.fc.weight"));
        }
    }

    @Test
    func should_remove_removed_canonical_names_from_every_lookup() throws {
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

        #expect(inventory.tensorCount() == 0);
        #expect(inventory.sourceIds().isEmpty);
    }
}
