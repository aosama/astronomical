import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * System telemetry journeys, migrating the telemetry half of
 * apps/supervisor/tests/rest_api/application/observatory_contracts.rs:
 * GET /v1/system/telemetry always answers with both fields present — the
 * GPU utilization a number in 0-100 or null on machines without an AGX
 * accelerator, and the memory pressure one of its level names or null —
 * and the kernel's pressure bitmask maps with the worst present level
 * winning and unknown values staying absent.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestSystemTelemetryTests {

    @Test
    func should_expose_gpu_utilization_and_memory_pressure_through_system_telemetry() throws {
        let telemetryJourney: SystemTelemetryJourney = try SystemTelemetryJourney.launch();
        defer { telemetryJourney.dispose() }

        let telemetryResponse: RestHttpResponse = try telemetryJourney.getSystemTelemetry();

        #expect(telemetryResponse.statusCode == 200);
        let telemetryDocument: [String: Any] = try ConfigReloadJourney.decodeObject(telemetryResponse);
        let gpuUtilization: Any? = telemetryDocument["gpu_utilization_percentage"];
        if gpuUtilization is NSNumber {
            let gpuPercentage: Double = (gpuUtilization as! NSNumber).doubleValue;
            #expect(gpuPercentage >= 0 && gpuPercentage <= 100);
        } else {
            #expect(gpuUtilization is NSNull,
                "gpu_utilization_percentage must be a number or null");
        }
        let memoryPressure: Any? = telemetryDocument["memory_pressure"];
        if memoryPressure is String {
            #expect(
                ["normal", "warning", "critical"].contains(memoryPressure as! String),
                "memory_pressure must be null, normal, warning, or critical");
        } else {
            #expect(memoryPressure is NSNull,
                "memory_pressure must be null, normal, warning, or critical");
        }
    }

    @Test
    func should_parse_macos_memory_pressure_bitmasks_without_treating_unknown_values_as_normal() throws {
        let memoryPressureCases: Array<(String, String?)> = [
            ("1", "normal"),
            ("2", "warning"),
            ("4", "critical"),
            ("6", "critical"),
            ("3", "warning"),
            ("0", nil),
            ("8", nil),
            ("not-a-pressure-level", nil),
        ];

        for (sysctlValueText, expectedMemoryPressureLevel) in memoryPressureCases {
            #expect(
                SystemTelemetry.parseMacOSMemoryPressureLevel(sysctlValueText)
                    == expectedMemoryPressureLevel,
                Comment(rawValue: "unexpected pressure mapping for \(sysctlValueText)"));
        }
    }
}

/// One telemetry journey: the serving route table over a fresh health state.
final class SystemTelemetryJourney {

    let routeTable: RestRouteTable;
    private let homeDirectoryUrl: URL;

    private init(routeTable: RestRouteTable, homeDirectoryUrl: URL) {
        self.routeTable = routeTable;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch() throws -> SystemTelemetryJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-telemetry-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity());
        return SystemTelemetryJourney(routeTable: routeTable, homeDirectoryUrl: homeDirectoryUrl);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func getSystemTelemetry() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: "GET",
            path: "/v1/system/telemetry");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw SystemTelemetryJourneyFailure.routeMissing;
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/system/telemetry"));
    }
}

/// Typed failures of the telemetry journeys.
enum SystemTelemetryJourneyFailure: Error {

    case routeMissing;
}
