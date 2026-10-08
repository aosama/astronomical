import Foundation

import Testing

import ModelServing

/// Forced capability-verdict injection seam contracts, port of
/// crates/model-serving/tests/hermetic/kernel_capability/forced_verdicts.rs.
///
/// CI hardware cannot make a real kernel fail, so the capability owner
/// accepts forced verdicts through this documented test-only constructor.
/// Production code must never call it.
@Suite
final class KernelCapabilityForcedVerdictsTests {

    @Test
    func shouldForceAnUnsupportedVerdictForASingleFamily() {
        let capabilities = WorkerKernelCapabilities.withForcedVerdictsForTests([
            (
                family: .sortedExpertWeightedSum,
                verdict: .unsupported(.outputMismatch(
                    description: "forced test verdict"))
            ),
        ]);

        #expect(
            capabilities.verdict(.sortedExpertWeightedSum)
                == .unsupported(.outputMismatch(description: "forced test verdict")));
        #expect(
            capabilities.verdict(.gatedDeltaSequence) == .unsupported(.unprobed),
            "families absent from the forced set stay fail-closed");
    }

    @Test
    func shouldForceASupportedVerdict() {
        let capabilities = WorkerKernelCapabilities.withForcedVerdictsForTests([
            (family: .gatedDeltaSequence, verdict: .supported),
        ]);

        #expect(capabilities.isCustomKernelSupported(.gatedDeltaSequence));
    }

    @Test
    func shouldHonorEveryForcedReasonForEveryFamily() {
        // Families are data, not tests: one loop proves the fail-closed
        // verdict contract across the whole family catalogue, so a new family
        // is covered by adding an enum variant, not a new test.
        let reasonCases: [KernelUnsupportedReason] = [
            .compilation(description: "forced compilation failure"),
            .execution(description: "forced execution failure"),
            .outputMismatch(description: "forced output mismatch"),
        ];
        let everyFamily: [CustomMetalKernelFamily] = [
            .sortedExpertWeightedSum,
            .fusedQuantizedExpertDecode,
            .gatedDeltaSequence,
            .gatedDeltaBoundaryCheckpoint,
        ];

        for unsupportedReason in reasonCases {
            let forcedVerdicts = everyFamily.map { family -> (
                family: CustomMetalKernelFamily,
                verdict: CustomKernelVerdict
            ) in
                (family: family, verdict: .unsupported(unsupportedReason));
            };
            let capabilities = WorkerKernelCapabilities.withForcedVerdictsForTests(
                forcedVerdicts);

            for family in everyFamily {
                #expect(
                    capabilities.verdict(family) == .unsupported(unsupportedReason),
                    "family \(family) must honor the forced \(unsupportedReason) verdict");
                #expect(
                    !capabilities.isCustomKernelSupported(family),
                    "a forced-unsupported family must never dispatch the custom kernel");
            }
        }
    }
}
