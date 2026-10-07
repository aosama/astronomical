import Foundation;

import IOKit;

import IpcProtocol;

/// Local macOS system telemetry exposed by the supervisor, porting
/// apps/supervisor/src/system_telemetry.rs: the GPU utilization the menu
/// bar samples directly from IOKit, and the kernel's memory-pressure level
/// read through sysctl. Both observations degrade to nil — the document's
/// fields stay present so clients never see missing keys.
public enum SystemTelemetry {

    static let sysctlExecutablePath: String = "/usr/sbin/sysctl";
    static let memoryPressureSysctlKey: String = "kern.memorystatus_vm_pressure_level";
    static let sysctlSampleTimeoutSeconds: TimeInterval = 2;
    static let memoryPressureNormalBit: UInt32 = 1;
    static let memoryPressureWarningBit: UInt32 = 2;
    static let memoryPressureCriticalBit: UInt32 = 4;
    static let memoryPressureKnownBits: UInt32 =
        memoryPressureNormalBit | memoryPressureWarningBit | memoryPressureCriticalBit;

    /// The telemetry document served by GET /v1/system/telemetry.
    public struct SystemTelemetryDocument {

        public var gpuUtilizationPercentage: Double?;
        public var memoryPressureLevel: String?;

        public init(
            gpuUtilizationPercentage: Double?,
            memoryPressureLevel: String?
        ) {
            self.gpuUtilizationPercentage = gpuUtilizationPercentage;
            self.memoryPressureLevel = memoryPressureLevel;
        }

        public func wireValue() -> JsonWireValue {
            var telemetryObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
            telemetryObject.appendEntry(
                key: "gpu_utilization_percentage",
                value: gpuUtilizationPercentage.map({ (utilization: Double) -> JsonWireValue in
                    return .double(utilization);
                }) ?? .null);
            telemetryObject.appendEntry(
                key: "memory_pressure",
                value: memoryPressureLevel.map({ (pressureLevel: String) -> JsonWireValue in
                    return .string(pressureLevel);
                }) ?? .null);
            return .object(telemetryObject);
        }
    }

    /// Samples both observations for one telemetry request.
    public static func sampleDocument() -> SystemTelemetryDocument {
        return SystemTelemetryDocument(
            gpuUtilizationPercentage: SystemTelemetry.sampleGpuUtilizationPercentage(),
            memoryPressureLevel: SystemTelemetry.sampleMemoryPressureLevel());
    }

    /// Maps the kernel's memory-pressure bitmask to its level name, mirroring
    /// parse_macos_memory_pressure_level: zero, unknown bits, and unparsable
    /// text are absent, and the worst present level wins.
    public static func parseMacOSMemoryPressureLevel(_ sysctlValueText: String) -> String? {
        let memoryPressureBitmask: UInt32? = UInt32(sysctlValueText.trimmingCharacters(in: .whitespacesAndNewlines));
        guard let memoryPressureBitmask = memoryPressureBitmask else {
            return nil;
        }
        if memoryPressureBitmask == 0 || memoryPressureBitmask & ~SystemTelemetry.memoryPressureKnownBits != 0 {
            return nil;
        }
        if memoryPressureBitmask & SystemTelemetry.memoryPressureCriticalBit != 0 {
            return "critical";
        }
        if memoryPressureBitmask & SystemTelemetry.memoryPressureWarningBit != 0 {
            return "warning";
        }
        if memoryPressureBitmask & SystemTelemetry.memoryPressureNormalBit != 0 {
            return "normal";
        }
        return nil;
    }

    /// Reads the kernel's memory-pressure level through sysctl with a
    /// bounded wait; any failure or timeout reports absent.
    public static func sampleMemoryPressureLevel() -> String? {
        let sysctlProcess: Process = Process();
        sysctlProcess.executableURL = URL(fileURLWithPath: SystemTelemetry.sysctlExecutablePath);
        sysctlProcess.arguments = ["-n", SystemTelemetry.memoryPressureSysctlKey];
        let standardOutputPipe: Pipe = Pipe();
        sysctlProcess.standardOutput = standardOutputPipe;
        sysctlProcess.standardError = Pipe();
        do {
            try sysctlProcess.run();
        } catch {
            return nil;
        }
        let waitDeadline: Date = Date().addingTimeInterval(SystemTelemetry.sysctlSampleTimeoutSeconds);
        while sysctlProcess.isRunning && Date() < waitDeadline {
            Thread.sleep(forTimeInterval: 0.02);
        }
        if sysctlProcess.isRunning {
            sysctlProcess.terminate();
            return nil;
        }
        guard sysctlProcess.terminationStatus == 0 else {
            return nil;
        }
        let sysctlOutputData: Data = standardOutputPipe.fileHandleForReading.readDataToEndOfFile();
        return SystemTelemetry.parseMacOSMemoryPressureLevel(
            String(decoding: sysctlOutputData, as: UTF8.self));
    }

    /// Samples GPU utilization from the first AGX accelerator's performance
    /// statistics, mirroring the menu bar's sampler: the utilization is a
    /// 0–100 double, and machines without an AGX accelerator report absent.
    public static func sampleGpuUtilizationPercentage() -> Double? {
        var acceleratorIterator: io_iterator_t = 0;
        let matchingResult: kern_return_t = IOServiceGetMatchingServices(
            kIOMainPortDefault,
            IOServiceMatching("AGXAccelerator"),
            &acceleratorIterator);
        guard matchingResult == KERN_SUCCESS else {
            return nil;
        }
        defer {
            _ = IOObjectRelease(acceleratorIterator);
        }
        var acceleratorService: io_object_t = IOIteratorNext(acceleratorIterator);
        while acceleratorService != 0 {
            defer {
                _ = IOObjectRelease(acceleratorService);
                acceleratorService = IOIteratorNext(acceleratorIterator);
            }
            guard let performanceStatistics: [String: Any] =
                IORegistryEntryCreateCFProperty(
                    acceleratorService,
                    "PerformanceStatistics" as CFString,
                    kCFAllocatorDefault,
                    0)?.takeRetainedValue() as? [String: Any],
                let gpuUtilizationNumber: NSNumber =
                    performanceStatistics["Device Utilization %"] as? NSNumber
            else {
                continue;
            }
            return gpuUtilizationNumber.doubleValue;
        }
        return nil;
    }
}
