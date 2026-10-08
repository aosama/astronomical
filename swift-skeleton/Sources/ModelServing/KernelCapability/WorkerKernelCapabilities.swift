import Foundation;

/// The retained capability verdicts for one worker process, port of the Rust
/// `WorkerKernelCapabilities`.
///
/// Verdicts are computed once at construction — at first model load, when the
/// worker already holds its runtime — and reused by every subsequent request
/// and model swap. Verdicts are never persisted to disk: an operating-system
/// update can change what compiles and runs, so a disk cache is a staleness
/// hazard for a probe cost measured in milliseconds.
public struct WorkerKernelCapabilities: Equatable, Sendable {

    private var verdictsByFamily: [CustomMetalKernelFamily: CustomKernelVerdict];

    private init(verdictsByFamily: [CustomMetalKernelFamily: CustomKernelVerdict]) {
        self.verdictsByFamily = verdictsByFamily;
    }

    /// Probes every provided family exactly once and retains the verdicts.
    ///
    /// Each probe runs under one attribution operation so the probe cost on
    /// the model-load critical path is measurable.
    public static func probeCustomKernels(
        _ kernelProbes: [any CustomMetalKernelProbe],
        _ performanceAttribution: PerformanceAttribution
    ) -> WorkerKernelCapabilities {
        var probedVerdictsByFamily: [CustomMetalKernelFamily: CustomKernelVerdict] = [:];
        for kernelProbe in kernelProbes {
            let probeVerdict: CustomKernelVerdict = performanceAttribution.measureOperation(
                .customKernelCapabilityProbe,
                { (_: PerformanceAttribution) -> CustomKernelVerdict in
                    switch kernelProbe.probe() {
                    case .success:
                        return .supported;
                    case .failure(let capabilityError):
                        return .unsupported(capabilityError.unsupportedReason);
                    }
                });
            probedVerdictsByFamily[kernelProbe.family] = probeVerdict;
        }
        return WorkerKernelCapabilities(verdictsByFamily: probedVerdictsByFamily);
    }

    /// Returns the retained verdict for one family; unprobed families fail
    /// closed.
    public func verdict(_ family: CustomMetalKernelFamily) -> CustomKernelVerdict {
        verdictsByFamily[family] ?? .unsupported(.unprobed);
    }

    /// Returns whether production dispatch may use the custom kernel.
    public func isCustomKernelSupported(_ family: CustomMetalKernelFamily) -> Bool {
        verdict(family) == .supported;
    }

    /// Builds an owner from explicit verdicts. CI hardware cannot make a real
    /// kernel fail, so hermetic and direct-MLX journeys force verdicts through
    /// this documented test-only constructor; production must never call it.
    public static func withForcedVerdictsForTests(
        _ forcedVerdicts: [(
            family: CustomMetalKernelFamily,
            verdict: CustomKernelVerdict
        )]
    ) -> WorkerKernelCapabilities {
        WorkerKernelCapabilities(verdictsByFamily: Dictionary(
            uniqueKeysWithValues: forcedVerdicts));
    }
}
