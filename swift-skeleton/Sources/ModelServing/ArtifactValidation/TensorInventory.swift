import Foundation;

/// Opaque identity for one validated safetensors source, assigned by artifact
/// discovery. Port of crates/model-serving/src/artifact_validation/tensor_inventory.rs.
public struct TensorSourceId: Equatable, Hashable, Comparable, Sendable {
    public let sourceNumber: UInt32;

    /// Creates an opaque source identity assigned by artifact discovery.
    public init(sourceNumber: UInt32) {
        self.sourceNumber = sourceNumber;
    }

    public static func < (leftSourceId: TensorSourceId, rightSourceId: TensorSourceId) -> Bool {
        return leftSourceId.sourceNumber < rightSourceId.sourceNumber;
    }
}

/// Architecture-neutral semantic ownership of a tensor.
public enum TensorSemanticRole: Equatable, Sendable {
    case target;
    case vision;
}

/// Boundary that declared one tensor location.
public enum TensorDeclarationOrigin: Equatable, Sendable {
    case mainIndex;
    case architectureSidecar;
}

/// Canonical and physical identity for one validated tensor location.
public struct TensorLocation: Equatable, Sendable {
    public let canonicalName: String;
    public let storedName: String;
    public let sourceId: TensorSourceId;
    public let semanticRole: TensorSemanticRole;
    public let declarationOrigin: TensorDeclarationOrigin;

    /// Creates a tensor location after architecture-specific name parsing.
    public init(
        canonicalName: String, storedName: String, sourceId: TensorSourceId,
        semanticRole: TensorSemanticRole, declarationOrigin: TensorDeclarationOrigin) {
        self.canonicalName = canonicalName;
        self.storedName = storedName;
        self.sourceId = sourceId;
        self.semanticRole = semanticRole;
        self.declarationOrigin = declarationOrigin;
    }
}

/// One physical (source, stored-name) pair that must map to exactly one
/// canonical tensor.
private struct PhysicalTensorLocation: Equatable, Hashable {
    let sourceId: TensorSourceId;
    let storedName: String;
}

/// Inventory ambiguity detected before runtime tensor allocation.
public enum TensorInventoryError: Error, Equatable {
    case canonicalNameCollision(canonicalName: String);
    case physicalLocationCollision(sourceId: TensorSourceId, storedName: String);

    public var errorDescription: String? {
        switch self {
        case .canonicalNameCollision(let canonicalName):
            return "canonical tensor name collision for \(canonicalName)";
        case .physicalLocationCollision(_, let storedName):
            return "physical tensor location collision for \(storedName)";
        }
    }
}

/// Convention-neutral canonical inventory for validated tensor locations.
/// Rust orders lookups and iteration through B-tree maps; this port preserves
/// the same deterministic canonical-name and source-number ordering.
public final class TensorInventory {
    private var locationsByCanonicalName: Dictionary<String, TensorLocation>;
    private var canonicalNameByPhysicalLocation: Dictionary<PhysicalTensorLocation, String>;

    public init() {
        self.locationsByCanonicalName = Dictionary();
        self.canonicalNameByPhysicalLocation = Dictionary();
    }

    /// Adds one location while rejecting canonical and physical ambiguity.
    public func insert(location: TensorLocation) throws -> Void {
        if self.locationsByCanonicalName[location.canonicalName] != nil {
            throw TensorInventoryError.canonicalNameCollision(canonicalName: location.canonicalName);
        }
        let physicalLocation: PhysicalTensorLocation = PhysicalTensorLocation(
            sourceId: location.sourceId, storedName: location.storedName);
        if self.canonicalNameByPhysicalLocation[physicalLocation] != nil {
            throw TensorInventoryError.physicalLocationCollision(
                sourceId: location.sourceId, storedName: location.storedName);
        }
        self.canonicalNameByPhysicalLocation[physicalLocation] = location.canonicalName;
        self.locationsByCanonicalName[location.canonicalName] = location;
    }

    public func location(canonicalName: String) -> TensorLocation? {
        return self.locationsByCanonicalName[canonicalName];
    }

    /// Locations in canonical-name order, mirroring the Rust B-tree iteration.
    public func locations() -> Array<TensorLocation> {
        return self.locationsByCanonicalName.keys.sorted().compactMap({ (canonicalName: String) -> TensorLocation? in
            return self.locationsByCanonicalName[canonicalName];
        });
    }

    /// Distinct source identities in ascending source-number order.
    public func sourceIds() -> Array<TensorSourceId> {
        var distinctSourceIds: Set<TensorSourceId> = Set();
        for tensorLocation: TensorLocation in self.locationsByCanonicalName.values {
            distinctSourceIds.insert(tensorLocation.sourceId);
        }
        return distinctSourceIds.sorted();
    }

    public func tensorCount() -> Int {
        return self.locationsByCanonicalName.count;
    }

    /// Removes locations by canonical name. Streaming revisions carry expert
    /// tensors in per-expert pack files instead of the indexed weight sources;
    /// stripping their locations keeps source validation and binding honest
    /// without weakening the remaining inventory contracts.
    public func removeCanonicalNames(canonicalNames: Set<String>) -> Void {
        self.locationsByCanonicalName = self.locationsByCanonicalName.filter({ (canonicalName: String, _: TensorLocation) -> Bool in
            return canonicalNames.contains(canonicalName) == false;
        });
        self.canonicalNameByPhysicalLocation = self.canonicalNameByPhysicalLocation.filter({ (_: PhysicalTensorLocation, canonicalName: String) -> Bool in
            return self.locationsByCanonicalName[canonicalName] != nil;
        });
    }
}
