import Foundation

/// Builds one selected expert page from startup-validated layer metadata.
/// Port of the Rust `build_quantized_expert_page_manifest_from_plan` and
/// `build_page_slot_by_global_expert_id`.
public enum QuantizedExpertPageManifestBuilder {

    /**
     * Builds a selected page: validated expert ids, per-shard bounded read
     * plans, and the compact slot lookup gathered projections consume.
     *
     * - Parameters:
     *   - layerPlan: The startup-validated layer geometry.
     *   - expertIds: The requested expert ids, validated ascending unique.
     * - Returns: The complete page manifest.
     */
    public static func buildPageManifest(
        layerPlan: QuantizedExpertLayerPlan,
        expertIds: [Int]
    ) throws -> QuantizedExpertPageManifest {
        let normalizedExpertIds: [Int] = try QuantizedExpertManifestValidation
            .validatedExpertIds(expertIds: expertIds, expertCapacity: layerPlan.expertCapacity)
        let sourceManifests: [QuantizedExpertShardManifest] = try
            QuantizedExpertSourceManifestBuilder.buildSourceManifests(
                tensorSources: layerPlan.tensorSources,
                selectedExpertIds: normalizedExpertIds)
        var payloadByteCount: UInt64 = 0
        for sourceManifest: QuantizedExpertShardManifest in sourceManifests {
            payloadByteCount += sourceManifest.payloadByteCount
        }
        let pageSlotByGlobalExpertId: [UInt32] =
            QuantizedExpertPageManifestBuilder.pageSlotByGlobalExpertId(
                normalizedExpertIds: normalizedExpertIds,
                expertCapacity: layerPlan.expertCapacity)
        return QuantizedExpertPageManifest(
            expertIds: normalizedExpertIds,
            pageSlotByGlobalExpertId: pageSlotByGlobalExpertId,
            sourceManifests: sourceManifests,
            payloadByteCount: payloadByteCount)
    }

    /**
     * Dense expert-id to compact-slot lookup with the absent sentinel. The
     * slot order matches the page tensors' first dimension, so gathered
     * expert work indexes rows without host-side remapping.
     */
    static func pageSlotByGlobalExpertId(
        normalizedExpertIds: [Int],
        expertCapacity: Int
    ) -> [UInt32] {
        var pageSlotByGlobalExpertId: [UInt32] = Array(
            repeating: QuantizedExpertPageManifest.ABSENT_PAGE_SLOT,
            count: expertCapacity)
        for (pageSlot, globalExpertId): (Int, Int) in normalizedExpertIds.enumerated() {
            pageSlotByGlobalExpertId[globalExpertId] = UInt32(pageSlot)
        }
        return pageSlotByGlobalExpertId
    }
}
