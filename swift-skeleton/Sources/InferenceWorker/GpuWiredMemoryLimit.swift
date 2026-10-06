import Foundation;

#if canImport(Glibc)
import Glibc;
#else
import Darwin;
#endif

import RuntimeIntegration;

/// The macOS GPU wired-memory policy the worker's MLX ceilings derive from.
///
/// Mirrors apps/inference-worker/src/worker_startup_gpu_memory.rs: parse the
/// `iogpu.wired_limit_mb` sysctl, treat zero as the system-default policy
/// sentinel rather than a zero-byte limit, and resolve the effective MLX
/// ceiling without ever exceeding the machine ceiling.
public enum GpuWiredMemoryLimit {

    private static let bytesPerMebibyte: UInt64 = 1024 * 1024;
    private static let iogpuWiredLimitSysctlKey: String = "iogpu.wired_limit_mb";
    private static let sysctlSampleTimeoutSeconds: TimeInterval = 2;
    private static let sysctlExecutablePath: String = "/usr/sbin/sysctl";

    /// The wired-memory policy reported by the sysctl.
    public enum GpuWiredMemoryLimitSetting: Equatable {
        /// A positive ceiling explicitly configured in mebibytes through sysctl.
        case explicitLimitBytes(Int);
        /// The sysctl reported zero, which is a policy sentinel rather than a
        /// zero-byte limit.
        case systemDefaultPolicy;
    }

    /// Parses the machine GPU wired-memory sysctl output.
    public static func parseIogpuWiredLimitSetting(
        _ wiredLimitMebibytesText: String
    ) throws -> GpuWiredMemoryLimitSetting {
        let trimmedText: String = wiredLimitMebibytesText.trimmingCharacters(in: .whitespacesAndNewlines);
        guard let wiredLimitMebibytes: UInt64 = UInt64(trimmedText) else {
            throw WorkerStartupError.invalidGpuWiredMemoryLimit(
                description: "wired-memory limit is not an unsigned integer");
        }
        if wiredLimitMebibytes == 0 {
            return .systemDefaultPolicy;
        }
        let (wiredLimitBytes, multiplicationOverflowed): (UInt64, Bool) =
            wiredLimitMebibytes.multipliedReportingOverflow(by: GpuWiredMemoryLimit.bytesPerMebibyte);
        if multiplicationOverflowed {
            throw WorkerStartupError.invalidGpuWiredMemoryLimit(
                description: "wired-memory limit exceeds the byte range");
        }
        guard let explicitLimitBytes: Int = Int(exactly: wiredLimitBytes) else {
            throw WorkerStartupError.invalidGpuWiredMemoryLimit(
                description: "wired-memory limit exceeds the platform integer range");
        }
        return .explicitLimitBytes(explicitLimitBytes);
    }

    /// Derives equal MLX active-memory and allocator-cache limits from the
    /// system ceiling; the split policy lives here so engine slices and the
    /// startup emission cannot drift apart.
    public static func deriveMlxMemoryLimits(
        fromGpuWiredLimit gpuWiredMemoryLimitBytes: Int
    ) -> (activeMemoryBytes: Int, allocatorCacheMemoryBytes: Int) {
        return (gpuWiredMemoryLimitBytes, gpuWiredMemoryLimitBytes);
    }

    /// Resolves the effective MLX ceiling without exceeding the machine ceiling.
    public static func resolveEffectiveMlxMemoryCeilingBytes(
        configuredMlxMemoryCeilingBytes: UInt64?,
        machineMlxMemoryCeilingBytes: Int
    ) -> Int {
        let machineMlxMemoryCeilingBytesAsUInt64: UInt64 = UInt64(machineMlxMemoryCeilingBytes);
        switch configuredMlxMemoryCeilingBytes {
        case .some(let configuredMlxMemoryCeilingBytes) where configuredMlxMemoryCeilingBytes < machineMlxMemoryCeilingBytesAsUInt64:
            return Int(configuredMlxMemoryCeilingBytes);
        default:
            return machineMlxMemoryCeilingBytes;
        }
    }

    /// Resolves the machine-specific GPU wired-memory ceiling without changing
    /// it: the sysctl sample when explicitly configured, otherwise the
    /// recommended working set the GPU stack reports for this machine.
    public static func sampleIogpuWiredLimitBytes() throws -> Int {
        let wiredLimitMebibytesText: String = try GpuWiredMemoryLimit.readSysctlWiredLimitMebibytes();
        switch try GpuWiredMemoryLimit.parseIogpuWiredLimitSetting(wiredLimitMebibytesText) {
        case let .explicitLimitBytes(explicitLimitBytes):
            return explicitLimitBytes;
        case .systemDefaultPolicy:
            do {
                return try MlxRuntime.maximumRecommendedGpuWorkingSetSizeBytes();
            } catch {
                throw WorkerStartupError.readMlxRecommendedGpuWorkingSet(
                    description: "could not read the recommended GPU working set: \(error)");
            }
        }
    }

    /// Runs the bounded sysctl sample. A wedged sysctl is killed at the
    /// deadline; a missing or failing sysctl is a typed startup failure.
    private static func readSysctlWiredLimitMebibytes() throws -> String {
        let sysctlProcess: Process = Process();
        sysctlProcess.executableURL = URL(fileURLWithPath: GpuWiredMemoryLimit.sysctlExecutablePath);
        sysctlProcess.arguments = ["-n", GpuWiredMemoryLimit.iogpuWiredLimitSysctlKey];
        let standardOutputPipe: Pipe = Pipe();
        sysctlProcess.standardOutput = standardOutputPipe;
        sysctlProcess.standardError = Pipe();
        do {
            try sysctlProcess.run();
        } catch {
            throw WorkerStartupError.sampleGpuWiredMemoryLimit(
                description: "could not run \(GpuWiredMemoryLimit.sysctlExecutablePath): \(error)");
        }
        let sampleDeadline: Date = Date().addingTimeInterval(
            GpuWiredMemoryLimit.sysctlSampleTimeoutSeconds);
        while sysctlProcess.isRunning && Date() < sampleDeadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        if sysctlProcess.isRunning {
            kill(Int32(sysctlProcess.processIdentifier), SIGKILL);
            sysctlProcess.waitUntilExit();
            throw WorkerStartupError.gpuWiredMemoryLimitSampleTimedOut;
        }
        guard sysctlProcess.terminationStatus == 0 else {
            throw WorkerStartupError.gpuWiredMemoryLimitSampleFailed;
        }
        let standardOutputData: Data = standardOutputPipe.fileHandleForReading.readDataToEndOfFile();
        guard let wiredLimitMebibytesText: String = String(data: standardOutputData, encoding: .utf8) else {
            throw WorkerStartupError.sampleGpuWiredMemoryLimit(
                description: "the wired-memory sysctl output is not valid UTF-8");
        }
        return wiredLimitMebibytesText;
    }
}
