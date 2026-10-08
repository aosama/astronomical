import Foundation

import Testing

import ModelServing

/// Behavior coverage for validated decoder-cache layouts and their
/// persistence/geometry arithmetic, port of
/// crates/model-serving/tests/hermetic/decoder_cache.rs.
@Suite
final class DecoderCacheLayoutTests {

    private static func mixedDecoderCacheLayout() throws -> DecoderCacheLayout {
        return try DecoderCacheLayout(layers: [
            .recurrentTensor(tensor: .fixed(
                tensorRoleName: "linear.convolution",
                dtype: .float16,
                dimensions: [1, 3, 8])),
            .composite(components: [
                .appendOnlyAttention(
                    keys: .sequence(
                        tensorRoleName: "attention.keys",
                        dtype: .bfloat16,
                        dimensions: [1, 2, 0, 4],
                        sequenceAxis: 2),
                    values: .sequence(
                        tensorRoleName: "attention.values",
                        dtype: .bfloat16,
                        dimensions: [1, 2, 0, 4],
                        sequenceAxis: 2),
                    capacityGrowthTokens: 256),
                .recurrentTensor(tensor: .fixed(
                    tensorRoleName: "linear.recurrent",
                    dtype: .float32,
                    dimensions: [1, 2, 4, 4])),
            ]),
        ])
    }

    @Test
    func shouldAcceptAMixedArchitectureNeutralDecoderCacheLayout() throws {
        let decoderCacheLayout = try Self.mixedDecoderCacheLayout()

        #expect(decoderCacheLayout.layerCount == 2)
        #expect(decoderCacheLayout.sequenceTensorCount == 2)
        #expect(decoderCacheLayout.boundaryTensorCount == 2)
        #expect(decoderCacheLayout.hasSequenceState)
        #expect(decoderCacheLayout.hasBoundaryState)
        #expect(try decoderCacheLayout.sequenceStatePayloadByteCountPerToken() == 32)
        #expect(
            try decoderCacheLayout.maximumSequenceTensorPayloadByteCount(sequenceTokenCount: 128)
                == 2_048)
        #expect(try decoderCacheLayout.persistenceAlignmentTokenCount() == 256)
        #expect(try decoderCacheLayout.boundarySnapshotPayloadByteCount() == 176)
        #expect(
            decoderCacheLayout.sequenceTensorLayouts().map(\.persistentTensorName)
                == ["layer_1_attention.keys", "layer_1_attention.values"])
        #expect(
            decoderCacheLayout.boundaryTensorLayouts().map(\.persistentTensorName)
                == ["layer_0_linear.convolution", "layer_1_linear.recurrent"])
    }

    @Test
    func shouldBoundIncrementalRestoreWorkspaceByOneBlockOrBoundarySnapshot() throws {
        let decoderCacheLayout = try Self.mixedDecoderCacheLayout()

        #expect(
            try decoderCacheLayout.incrementalRestoreSourceWorkspaceByteCount(
                sequenceBlockTokenCount: 2) == 176,
            "the complete boundary snapshot must remain budgeted when larger than one sequence block")
        #expect(
            try decoderCacheLayout.incrementalRestoreSourceWorkspaceByteCount(
                sequenceBlockTokenCount: 16) == 512,
            "sequence source admission should scale with one block, not the restored prefix")
    }

    @Test
    func shouldDeriveTheLeastCommonPersistenceAlignmentForMixedAttentionGrowth() throws {
        let decoderCacheLayout = try DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "first.keys",
                    dtype: .float16,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "first.values",
                    dtype: .float16,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                capacityGrowthTokens: 6),
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "second.keys",
                    dtype: .float32,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "second.values",
                    dtype: .float32,
                    dimensions: [1, 0, 2],
                    sequenceAxis: 1),
                capacityGrowthTokens: 8),
        ])

        #expect(try decoderCacheLayout.persistenceAlignmentTokenCount() == 24)
    }

    @Test
    func shouldRejectPersistenceAlignmentOverflow() throws {
        let decoderCacheLayout = try DecoderCacheLayout(layers: [
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "first.keys",
                    dtype: .float16,
                    dimensions: [1, 0],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "first.values",
                    dtype: .float16,
                    dimensions: [1, 0],
                    sequenceAxis: 1),
                capacityGrowthTokens: Int.max),
            .appendOnlyAttention(
                keys: .sequence(
                    tensorRoleName: "second.keys",
                    dtype: .float16,
                    dimensions: [1, 0],
                    sequenceAxis: 1),
                values: .sequence(
                    tensorRoleName: "second.values",
                    dtype: .float16,
                    dimensions: [1, 0],
                    sequenceAxis: 1),
                capacityGrowthTokens: Int.max - 1),
        ])

        #expect(throws: DecoderCacheLayoutError.persistenceAlignmentTokenCountOverflow) {
            try decoderCacheLayout.persistenceAlignmentTokenCount()
        }
    }

    @Test
    func shouldRejectDuplicateTensorRolesAcrossOneLayer() {
        #expect(
            throws: DecoderCacheLayoutError.duplicateTensorRole(
                layerIndex: 0, tensorRoleName: "state")
        ) {
            try DecoderCacheLayout(layers: [
                .composite(components: [
                    .recurrentTensor(tensor: .fixed(
                        tensorRoleName: "state",
                        dtype: .float32,
                        dimensions: [1, 4])),
                    .recurrentTensor(tensor: .fixed(
                        tensorRoleName: "state",
                        dtype: .float32,
                        dimensions: [1, 4])),
                ]),
            ])
        }
    }

    @Test
    func shouldRejectASequenceAxisOutsideTheTensorRank() {
        #expect(
            throws: DecoderCacheLayoutError.sequenceAxisOutsideTensorRank(
                layerIndex: 0,
                tensorRoleName: "attention.keys",
                sequenceAxis: 4,
                tensorRank: 4)
        ) {
            try DecoderCacheLayout(layers: [
                .appendOnlyAttention(
                    keys: .sequence(
                        tensorRoleName: "attention.keys",
                        dtype: .bfloat16,
                        dimensions: [1, 2, 0, 4],
                        sequenceAxis: 4),
                    values: .sequence(
                        tensorRoleName: "attention.values",
                        dtype: .bfloat16,
                        dimensions: [1, 2, 0, 4],
                        sequenceAxis: 2),
                    capacityGrowthTokens: 256),
            ])
        }
    }
}
