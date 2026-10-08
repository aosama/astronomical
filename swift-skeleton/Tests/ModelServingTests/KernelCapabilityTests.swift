import Foundation

import Testing

import ModelServing

/// Fake probe counting its invocations so the once-per-worker-process
/// contract is observable without a GPU.
private final class CountingProbe: CustomMetalKernelProbe {

    let family: CustomMetalKernelFamily;
    private let probeOutcome: Result<Void, KernelCapabilityError>;
    private(set) var invocationCount: Int = 0;

    init(
        family: CustomMetalKernelFamily,
        outcome: Result<Void, KernelCapabilityError>
    ) {
        self.family = family;
        self.probeOutcome = outcome;
    }

    func probe() -> Result<Void, KernelCapabilityError> {
        invocationCount += 1;
        return probeOutcome;
    }
}

private extension Result where Failure == KernelCapabilityError {

    var isSuccess: Bool {
        if case .success = self {
            return true;
        }
        return false;
    }

    var isOutputMismatch: Bool {
        if case .failure(.outputMismatch(let description)) = self {
            return !description.isEmpty;
        }
        return false;
    }
}

private func supportedProbePair(
    _ family: CustomMetalKernelFamily
) -> (probe: CountingProbe, invocationCount: () -> Int) {
    let countingProbe = CountingProbe(family: family, outcome: .success(()));
    return (countingProbe, { countingProbe.invocationCount });
}

/// Per-worker custom-Metal-kernel capability owner contracts: probe outcomes
/// become verdicts, verdicts are reused without re-probing, and unprobed
/// families fail closed, port of
/// crates/model-serving/tests/hermetic/kernel_capability/mod.rs.
@Suite
final class KernelCapabilityTests {

    @Test
    func shouldProbeEveryFamilyOnceAndReuseVerdictsWithoutReprobing() {
        let (weightedSumProbe, weightedSumInvocations) = supportedProbePair(
            .sortedExpertWeightedSum);
        let (gatedDeltaProbe, gatedDeltaInvocations) = supportedProbePair(
            .gatedDeltaSequence);
        let performanceAttribution = PerformanceAttribution.disabled();

        let capabilities = WorkerKernelCapabilities.probeCustomKernels(
            [weightedSumProbe, gatedDeltaProbe],
            performanceAttribution);

        #expect(
            capabilities.verdict(.sortedExpertWeightedSum) == .supported);
        #expect(
            capabilities.verdict(.gatedDeltaSequence) == .supported);
        #expect(
            capabilities.isCustomKernelSupported(.sortedExpertWeightedSum));

        for _ in 0..<10 {
            _ = capabilities.verdict(.sortedExpertWeightedSum);
        }

        #expect(
            weightedSumInvocations() == 1,
            "verdict reads must never re-probe a supported family");
        #expect(
            gatedDeltaInvocations() == 1,
            "each family probes exactly once per worker process");
    }

    @Test
    func shouldReportATypedCompilationReasonWithoutLosingOtherFamilies() {
        let failingProbe = CountingProbe(
            family: .sortedExpertWeightedSum,
            outcome: .failure(.compilation(
                description: "the probe kernel source failed to compile")));
        let (supportedProbe, supportedInvocations) = supportedProbePair(.gdnDecodePrework);
        let performanceAttribution = PerformanceAttribution.disabled();

        let capabilities = WorkerKernelCapabilities.probeCustomKernels(
            [failingProbe, supportedProbe],
            performanceAttribution);

        #expect(
            capabilities.verdict(.sortedExpertWeightedSum)
                == .unsupported(.compilation(
                    description: "the probe kernel source failed to compile")));
        #expect(
            !capabilities.isCustomKernelSupported(.sortedExpertWeightedSum));
        #expect(
            capabilities.verdict(.gdnDecodePrework) == .supported,
            "one unsupported family must never demote an independently supported family");
        #expect(failingProbe.invocationCount == 1);
        #expect(supportedInvocations() == 1);
    }

    @Test
    func shouldDistinguishExecutionFailuresFromOutputMismatches() {
        let executionProbe = CountingProbe(
            family: .gatedDeltaSequence,
            outcome: .failure(.execution(
                description: "the bounded probe launch failed")));
        let mismatchProbe = CountingProbe(
            family: .fusedQuantizedExpertDecode,
            outcome: .failure(.outputMismatch(
                description: "probe output value 0 read 0.000000 but expected 133.000000")));
        let performanceAttribution = PerformanceAttribution.disabled();

        let capabilities = WorkerKernelCapabilities.probeCustomKernels(
            [executionProbe, mismatchProbe],
            performanceAttribution);

        #expect(
            capabilities.verdict(.gatedDeltaSequence)
                == .unsupported(.execution(
                    description: "the bounded probe launch failed")));
        #expect(
            capabilities.verdict(.fusedQuantizedExpertDecode)
                == .unsupported(.outputMismatch(
                    description: "probe output value 0 read 0.000000 but expected 133.000000")));
    }

    @Test
    func shouldFailClosedForAFamilyThatWasNeverProbed() {
        let (weightedSumProbe, _) = supportedProbePair(.sortedExpertWeightedSum);
        let performanceAttribution = PerformanceAttribution.disabled();

        let capabilities = WorkerKernelCapabilities.probeCustomKernels(
            [weightedSumProbe],
            performanceAttribution);

        #expect(
            capabilities.verdict(.gatedDeltaBoundaryCheckpoint)
                == .unsupported(.unprobed),
            "an unprobed family must never be assumed supported");
    }

    @Test
    func shouldAcceptExactProbeOutputsAndRejectSilentZeroDispatches() {
        let expectedOutputs: [Float] = [133.0, 134.0, 120.0, 121.0];

        #expect(
            validateProbeOutputs(expectedOutputs, expectedOutputs).isSuccess,
            "exact expected outputs must validate");
        // A silently dropped Metal dispatch that returns zeros is a known
        // historical failure signature; the probe validator must reject it.
        #expect(
            validateProbeOutputs([0.0, 0.0, 0.0, 0.0], expectedOutputs).isOutputMismatch,
            "an all-zeros dispatch is a known silent-failure signature and must be rejected");
        #expect(
            validateProbeOutputs([133.0, 134.0], expectedOutputs).isOutputMismatch,
            "a shortened output must be rejected as an output mismatch");
    }
}
