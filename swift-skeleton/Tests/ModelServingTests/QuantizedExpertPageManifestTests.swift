import Foundation

import Testing

import ModelServing

/// Hermetic journeys over the quantized expert page manifest, port of
/// crates/model-serving/tests/hermetic/quantized_expert_page_manifest.rs:
/// a retained page names exactly the routed experts it is missing,
/// distinguishes complete from partial pages, partitions one route's
/// assignments without duplicate execution, and the layer plan derives
/// exact per-expert and complete-layer payload bytes from tensor
/// geometry.
@Suite
final class QuantizedExpertPageManifestTests {

    private static func retainedPageManifest() -> QuantizedExpertPageManifest {
        return QuantizedExpertPageManifest(
            expertIds: [1, 3],
            pageSlotByGlobalExpertId: [
                QuantizedExpertPageManifest.ABSENT_PAGE_SLOT, 0,
                QuantizedExpertPageManifest.ABSENT_PAGE_SLOT, 1,
                QuantizedExpertPageManifest.ABSENT_PAGE_SLOT,
            ],
            sourceManifests: [],
            payloadByteCount: 200)
    }

    @Test
    func should_identify_only_routed_experts_missing_from_a_retained_page() {
        let retainedPageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifestTests.retainedPageManifest()

        #expect(retainedPageManifest.missingExpertIds(selectedExpertIds: [0, 1, 3, 4]) == [0, 4])
    }

    @Test
    func should_report_complete_route_coverage_for_a_retained_page() {
        let retainedPageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifestTests.retainedPageManifest()

        #expect(retainedPageManifest.containsEveryExpert(selectedExpertIds: [1, 3]))
        #expect(retainedPageManifest.containsEveryExpert(selectedExpertIds: [1, 2, 3]) == false)
    }

    @Test
    func should_distinguish_complete_and_partial_expert_pages() {
        let partialPageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifestTests.retainedPageManifest()
        let completePageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifest(
            expertIds: [0, 1, 2],
            pageSlotByGlobalExpertId: [0, 1, 2],
            sourceManifests: [],
            payloadByteCount: 300)

        #expect(partialPageManifest.containsAllExperts() == false)
        #expect(completePageManifest.containsAllExperts())
    }

    @Test
    func should_partition_route_assignments_without_duplicate_execution() {
        let retainedPageManifest: QuantizedExpertPageManifest = QuantizedExpertPageManifestTests.retainedPageManifest()

        let routePartition: ExpertRoutePartition = retainedPageManifest
            .partitionRouteAssignments(selectedExpertIds: [3, 0, 1, 4, 3, 1])

        #expect(routePartition.retainedAssignmentPositions == [0, 2, 4, 5])
        #expect(routePartition.retainedExpertIds == [1, 3])
        #expect(routePartition.missingAssignmentPositions == [1, 3])
        #expect(routePartition.missingExpertIds == [0, 4])
        #expect(
            routePartition.retainedAssignmentPositions.count + routePartition.missingAssignmentPositions.count
            == 6)
    }

    @Test
    func should_derive_exact_per_expert_and_complete_layer_payload_from_tensor_geometry() throws {
        let layerPlan: QuantizedExpertLayerPlan = QuantizedExpertLayerPlan(
            layerPrefix: "fictional.layers.0",
            tensorSources: [
                QuantizedExpertPageManifestTests.tensorSource(
                    tensorName: "gate.weight", bytesPerExpert: 10),
                QuantizedExpertPageManifestTests.tensorSource(
                    tensorName: "up.weight", bytesPerExpert: 5),
            ],
            expertCapacity: 4,
            quantizationBits: 4,
            quantizationGroupSize: 64,
            quantizationMode: .affine)

        #expect(try layerPlan.expertPayloadByteCount() == 15)
        #expect(try layerPlan.completeExpertPayloadByteCount() == 60)
    }

    private static func tensorSource(tensorName: String, bytesPerExpert: Int) -> QuantizedTensorSource {
        return QuantizedTensorSource(
            tensorName: tensorName,
            projectionName: "fictional_projection",
            parameterName: "weight",
            quantizationBits: 4,
            quantizationGroupSize: 64,
            sourceFileName: "fictional-model.safetensors",
            sourceFileSizeBytes: 1_000,
            dtype: .u32,
            fullShape: [4, 2, 2],
            tensorPayloadOffsetBytes: 0,
            bytesPerExpert: bytesPerExpert,
            expertCapacity: 4)
    }
}
