import Foundation

/// Compacts raw router selections into the stored route-history form, port
/// of the Rust `sorted_unique_layer_routed_expert_ids`.
public enum RouteObservationCompaction {

    /// Compacts one layer's raw routed expert identifiers into the sorted
    /// unique form the history stores. Returns `nil` when an identifier
    /// cannot fit the compact element, which would indicate a router
    /// contract violation rather than data to train on.
    public static func sortedUniqueLayerRoutedExpertIds(
        rawExpertIds: [UInt32]
    ) -> LayerRoutedExpertIds? {
        var compactedExpertIds: LayerRoutedExpertIds = []
        compactedExpertIds.reserveCapacity(rawExpertIds.count)
        for rawExpertId in rawExpertIds {
            guard let expertId = UInt16(exactly: rawExpertId) else {
                return nil
            }
            compactedExpertIds.append(expertId)
        }
        compactedExpertIds.sort()
        var uniqueExpertIds: LayerRoutedExpertIds = []
        for expertId in compactedExpertIds where uniqueExpertIds.last != expertId {
            uniqueExpertIds.append(expertId)
        }
        return uniqueExpertIds
    }
}
