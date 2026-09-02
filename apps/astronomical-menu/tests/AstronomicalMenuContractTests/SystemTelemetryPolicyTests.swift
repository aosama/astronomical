import Foundation
import XCTest

@testable import AstronomicalMenuCore

final class SystemTelemetryPolicyTests: XCTestCase {
  func test_should_sample_system_telemetry_only_for_a_visible_popover_or_active_request() {
    XCTAssertFalse(
      systemTelemetrySamplingIsRequired(popoverIsVisible: false, requestIsActive: false))
    XCTAssertTrue(systemTelemetrySamplingIsRequired(popoverIsVisible: true, requestIsActive: false))
    XCTAssertTrue(systemTelemetrySamplingIsRequired(popoverIsVisible: false, requestIsActive: true))
  }

  func test_should_limit_visible_system_telemetry_updates_to_once_per_second() {
    XCTAssertTrue(systemTelemetrySampleIsDue(elapsedSinceLastSample: nil))
    XCTAssertFalse(
      systemTelemetrySampleIsDue(elapsedSinceLastSample: .milliseconds(250))
    )
    XCTAssertTrue(
      systemTelemetrySampleIsDue(elapsedSinceLastSample: .seconds(1))
    )
  }

  func test_should_decode_verified_agx_gpu_utilization_without_an_unneeded_memory_counter() {
    let agxTelemetry = agxTelemetryFromPerformanceStatistics([
      "Device Utilization %": NSNumber(value: 17)
    ])

    XCTAssertEqual(agxTelemetry?.gpuUtilizationPercentage, 17)
  }

  func test_should_keep_unavailable_system_telemetry_distinct_from_normal_pressure() {
    XCTAssertEqual(
      SystemTelemetrySnapshot.unavailable.memoryPressureTitle,
      .unavailable
    )
  }
}
