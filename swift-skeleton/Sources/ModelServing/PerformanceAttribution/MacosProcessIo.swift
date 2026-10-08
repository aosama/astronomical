import Darwin;
import Foundation;

/// Typed current-process disk input/output evidence from macOS `proc_pid_rusage`.
///
/// The counters are process-attributed physical disk input/output according to
/// XNU accounting. They are wider than expert paging: any worker disk activity
/// between two snapshots can contribute to a delta.
public enum MacosProcessIo {

    /// Flavor 4 is old enough for every supported Astronomical macOS
    /// deployment and is the first stable layout that contains the cumulative
    /// disk read/write fields needed here.
    private static let resourceUsageFlavorV4: Int32 = 4;

    /// Samples cumulative physical disk input/output attributed to this process.
    ///
    /// This function performs no file input/output itself. The disabled
    /// performance path never calls it; enabled attribution samples only at
    /// report boundaries.
    public static func sampleCurrentProcessIo() -> Result<
        MacosProcessIoSnapshot,
        MacosProcessIoError
    > {
        var resourceUsage = MacosResourceUsageInfoV4();
        let samplingStatus = withUnsafeMutablePointer(to: &resourceUsage) {
            resourceUsagePointer -> Int32 in
            proc_pid_rusage(
                processId: getpid(),
                flavor: resourceUsageFlavorV4,
                buffer: UnsafeMutableRawPointer(resourceUsagePointer));
        };
        if samplingStatus != 0 {
            return .failure(
                .samplingFailed(
                    osErrorCode: Int32(errno)));
        }
        return .success(
            MacosProcessIoSnapshot.fromCumulativeBytes(
                physicalDiskReadBytes: resourceUsage.riDiskioBytesread,
                physicalDiskWrittenBytes: resourceUsage.riDiskioByteswritten));
    }
}

/// One cumulative process input/output sample.
public struct MacosProcessIoSnapshot: Equatable, Sendable {

    private let physicalDiskReadBytesValue: UInt64;
    private let physicalDiskWrittenBytesValue: UInt64;

    /// Builds one snapshot from the raw cumulative counters, exposed for the
    /// sampling and delta evidence journeys that must not depend on live
    /// operating-system counters.
    public static func fromCumulativeBytes(
        physicalDiskReadBytes: UInt64,
        physicalDiskWrittenBytes: UInt64
    ) -> MacosProcessIoSnapshot {
        MacosProcessIoSnapshot(
            physicalDiskReadBytesValue: physicalDiskReadBytes,
            physicalDiskWrittenBytesValue: physicalDiskWrittenBytes);
    }

    public var physicalDiskReadBytes: UInt64 {
        physicalDiskReadBytesValue;
    }

    public var physicalDiskWrittenBytes: UInt64 {
        physicalDiskWrittenBytesValue;
    }

    /// Computes one request/report interval from two cumulative process samples.
    ///
    /// A process restart, kernel anomaly, or future accounting reset can make a
    /// later counter smaller. Such a sample is unavailable evidence, not zero
    /// traffic and not unsigned wraparound, so subtraction fails explicitly.
    public func deltaSince(
        _ earlierSnapshot: MacosProcessIoSnapshot
    ) -> Result<MacosProcessIoDelta, MacosProcessIoError> {
        let (physicalDiskReadBytes, readUnderflow) = physicalDiskReadBytesValue
            .subtractingReportingOverflow(
                earlierSnapshot.physicalDiskReadBytes);
        if readUnderflow {
            return .failure(
                .counterRegressed(
                    counterName: "ri_diskio_bytesread",
                    earlierBytes: earlierSnapshot.physicalDiskReadBytes,
                    laterBytes: physicalDiskReadBytesValue));
        }
        let (physicalDiskWrittenBytes, writtenUnderflow) = physicalDiskWrittenBytesValue
            .subtractingReportingOverflow(
                earlierSnapshot.physicalDiskWrittenBytes);
        if writtenUnderflow {
            return .failure(
                .counterRegressed(
                    counterName: "ri_diskio_byteswritten",
                    earlierBytes: earlierSnapshot.physicalDiskWrittenBytes,
                    laterBytes: physicalDiskWrittenBytesValue));
        }
        return .success(
            MacosProcessIoDelta(
                physicalDiskReadBytes: physicalDiskReadBytes,
                physicalDiskWrittenBytes: physicalDiskWrittenBytes));
    }
}

/// Process-attributed physical disk input/output during one measured interval.
///
/// "Process-attributed" is deliberate: these values are not scoped to expert
/// files. Tokenizer, prompt-cache, logging, or unrelated worker reads during
/// the same report interval may contribute. They are still the correct
/// companion to logical read bytes because they reveal whether macOS needed
/// physical input/output at all.
public struct MacosProcessIoDelta: Equatable, Sendable {

    public let physicalDiskReadBytes: UInt64;
    public let physicalDiskWrittenBytes: UInt64;
}

/// A process input/output sample cannot be used as evidence.
public enum MacosProcessIoError: Error, Equatable, CustomStringConvertible, Sendable {

    case samplingFailed(osErrorCode: Int32)
    case counterRegressed(
        counterName: String,
        earlierBytes: UInt64,
        laterBytes: UInt64)
    case unsupportedPlatform

    public var description: String {
        switch self {
        case .samplingFailed(let osErrorCode):
            return "macOS proc_pid_rusage failed with operating-system error \(osErrorCode)";
        case .counterRegressed(
            let counterName,
            let earlierBytes,
            let laterBytes):
            return "macOS process I/O counter \(counterName) regressed from \(earlierBytes) to \(laterBytes) bytes";
        case .unsupportedPlatform:
            return "process I/O accounting is available only on macOS";
        }
    }
}

/// Exact transcription of Apple's public `rusage_info_v4` layout, owned here
/// rather than borrowed from the C headers so the flavor and the Swift layout
/// that receives it cannot drift apart. The `ri_uuid` bytes merge into two
/// 64-bit slots for layout only; they are never read. Fields following disk
/// input/output remain present so the allocation is large enough for the
/// selected flavor on every supported SDK/runtime pair.
struct MacosResourceUsageInfoV4 {

    var riUuidSlots: (UInt64, UInt64) = (0, 0);
    var riUserTime: UInt64 = 0;
    var riSystemTime: UInt64 = 0;
    var riPkgIdleWkups: UInt64 = 0;
    var riInterruptWkups: UInt64 = 0;
    var riPageins: UInt64 = 0;
    var riWiredSize: UInt64 = 0;
    var riResidentSize: UInt64 = 0;
    var riPhysFootprint: UInt64 = 0;
    var riProcStartAbstime: UInt64 = 0;
    var riProcExitAbstime: UInt64 = 0;
    var riChildUserTime: UInt64 = 0;
    var riChildSystemTime: UInt64 = 0;
    var riChildPkgIdleWkups: UInt64 = 0;
    var riChildInterruptWkups: UInt64 = 0;
    var riChildPageins: UInt64 = 0;
    var riChildElapsedAbstime: UInt64 = 0;
    var riDiskioBytesread: UInt64 = 0;
    var riDiskioByteswritten: UInt64 = 0;
    var riCpuTimeQosDefault: UInt64 = 0;
    var riCpuTimeQosMaintenance: UInt64 = 0;
    var riCpuTimeQosBackground: UInt64 = 0;
    var riCpuTimeQosUtility: UInt64 = 0;
    var riCpuTimeQosLegacy: UInt64 = 0;
    var riCpuTimeQosUserInitiated: UInt64 = 0;
    var riCpuTimeQosUserInteractive: UInt64 = 0;
    var riBilledSystemTime: UInt64 = 0;
    var riServicedSystemTime: UInt64 = 0;
    var riLogicalWrites: UInt64 = 0;
    var riLifetimeMaxPhysFootprint: UInt64 = 0;
    var riInstructions: UInt64 = 0;
    var riCycles: UInt64 = 0;
    var riBilledEnergy: UInt64 = 0;
    var riServicedEnergy: UInt64 = 0;
    var riIntervalMaxPhysFootprint: UInt64 = 0;
}

@_silgen_name("proc_pid_rusage")
private func proc_pid_rusage(
    processId: Int32,
    flavor: Int32,
    buffer: UnsafeMutableRawPointer
) -> Int32;
