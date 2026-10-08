import Foundation

import Testing

import ModelServing

/// Behavior coverage for rotating-cache admission geometry and its
/// boundary persistence, port of
/// crates/model-serving/tests/hermetic/attention/rotating_admission.rs.
@Suite
final class RotatingAdmissionTests {

    private static func rotatingLayout(windowSize: Int) -> DecoderCacheLayerLayout {
        return .rotatingWindowAttention(
            keys: .fixed(
                tensorRoleName: "attention.keys",
                dtype: .bfloat16,
                dimensions: [1, 2, 8, 4]),
            values: .fixed(
                tensorRoleName: "attention.values",
                dtype: .bfloat16,
                dimensions: [1, 2, 8, 4]),
            windowSize: windowSize)
    }

    @Test
    func shouldChargeWindowPlusChunkMinusOneForRotatingPrefill() throws {
        let transientPeakExpectations: [(windowSize: UInt32, chunkTokens: UInt32, peakTokens: UInt32)] = [
            (8, 1, 8),
            (8, 3, 10),
            (64, 16, 79),
            (512, 128, 639),
            (4, 0, 4),
        ]
        for transientPeakExpectation in transientPeakExpectations {
            #expect(
                try RotatingAdmission.rotatingPrefillTransientTokenCount(
                    windowSize: transientPeakExpectation.windowSize,
                    promptChunkTokenCount: transientPeakExpectation.chunkTokens)
                    == transientPeakExpectation.peakTokens)
        }
    }

    @Test
    func shouldBoundCommittedTokensToTheWindow() {
        #expect(RotatingAdmission.rotatingCommittedTokenCount(windowSize: 8, absolutePosition: 3) == 3)
        #expect(RotatingAdmission.rotatingCommittedTokenCount(windowSize: 8, absolutePosition: 40) == 8)
    }

    @Test
    func shouldValidateAndPersistRotatingBoundaryGeometry() throws {
        #expect(
            throws: DecoderCacheLayoutError.zeroRotatingWindowSize(layerIndex: 0)
        ) {
            try DecoderCacheLayout(layers: [Self.rotatingLayout(windowSize: 0)])
        }

        let layout = try DecoderCacheLayout(layers: [Self.rotatingLayout(windowSize: 8)])
        #expect(layout.sequenceTensorCount == 0)
        #expect(layout.boundaryTensorCount == 4)
        #expect(
            layout.boundaryTensorLayouts().map(\.persistentTensorName)
                == [
                    "layer_0_attention.keys",
                    "layer_0_attention.values",
                    "layer_0_attention.absolute_position",
                    "layer_0_attention.ring_write_index",
                ])
    }

    @Test
    func shouldRejectZeroWindowAndOverflowingTransientGeometry() {
        #expect(throws: RotatingAdmissionError.zeroWindowSize) {
            try RotatingAdmission.rotatingPrefillTransientTokenCount(
                windowSize: 0, promptChunkTokenCount: 4)
        }
        #expect(throws: RotatingAdmissionError.transientTokenCountOverflow(
            windowSize: UInt32.max, promptChunkTokenCount: 2)) {
            try RotatingAdmission.rotatingPrefillTransientTokenCount(
                windowSize: UInt32.max, promptChunkTokenCount: 2)
        }
    }
}
