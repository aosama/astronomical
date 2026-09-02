import Foundation
import IOKit

enum SystemMemoryPressureTitle: String {
  case normal = "Normal"
  case warning = "Warning"
  case critical = "Critical"
  case unavailable = "Unavailable"
}

struct SystemTelemetrySnapshot {
  let gpuUtilizationPercentage: Double?
  let memoryPressureTitle: SystemMemoryPressureTitle

  static let unavailable = SystemTelemetrySnapshot(
    gpuUtilizationPercentage: nil,
    memoryPressureTitle: .unavailable
  )
}

struct AGXTelemetry {
  let gpuUtilizationPercentage: Double
}

func agxTelemetryFromPerformanceStatistics(
  _ performanceStatistics: [String: Any]
) -> AGXTelemetry? {
  guard
    let gpuUtilizationNumber = performanceStatistics["Device Utilization %"] as? NSNumber
  else {
    return nil
  }
  let gpuUtilizationPercentage = gpuUtilizationNumber.doubleValue
  guard (0...100).contains(gpuUtilizationPercentage) else { return nil }
  return AGXTelemetry(
    gpuUtilizationPercentage: gpuUtilizationPercentage
  )
}

func systemTelemetrySamplingIsRequired(popoverIsVisible: Bool, requestIsActive: Bool) -> Bool {
  popoverIsVisible || requestIsActive
}

func systemTelemetrySampleIsDue(elapsedSinceLastSample: Duration?) -> Bool {
  guard let elapsedSinceLastSample else { return true }
  return elapsedSinceLastSample >= .seconds(1)
}

final class SystemTelemetrySampler: @unchecked Sendable {
  private let memoryPressureSource: DispatchSourceMemoryPressure
  private let stateLock = NSLock()
  private var memoryPressureTitle: SystemMemoryPressureTitle = .normal

  init() {
    memoryPressureSource = DispatchSource.makeMemoryPressureSource(
      eventMask: [.normal, .warning, .critical],
      queue: DispatchQueue(label: "astronomical-menu-memory-pressure")
    )
    memoryPressureSource.setEventHandler { [weak self] in
      guard let self else { return }
      let pressureEvent = memoryPressureSource.data
      let nextPressureTitle: SystemMemoryPressureTitle =
        if pressureEvent.contains(.critical) {
          .critical
        } else if pressureEvent.contains(.warning) {
          .warning
        } else {
          .normal
        }
      stateLock.withLock { memoryPressureTitle = nextPressureTitle }
    }
    memoryPressureSource.resume()
  }

  deinit { memoryPressureSource.cancel() }

  func sample() -> SystemTelemetrySnapshot {
    let agxTelemetry = readAGXTelemetry()
    return SystemTelemetrySnapshot(
      gpuUtilizationPercentage: agxTelemetry?.gpuUtilizationPercentage,
      memoryPressureTitle: stateLock.withLock { memoryPressureTitle }
    )
  }

  private func readAGXTelemetry() -> AGXTelemetry? {
    var acceleratorIterator: io_iterator_t = 0
    let matchingResult = IOServiceGetMatchingServices(
      kIOMainPortDefault,
      IOServiceMatching("AGXAccelerator"),
      &acceleratorIterator
    )
    guard matchingResult == KERN_SUCCESS else { return nil }
    defer { IOObjectRelease(acceleratorIterator) }

    var acceleratorService = IOIteratorNext(acceleratorIterator)
    while acceleratorService != 0 {
      defer {
        IOObjectRelease(acceleratorService)
        acceleratorService = IOIteratorNext(acceleratorIterator)
      }
      guard
        let performanceStatistics = IORegistryEntryCreateCFProperty(
          acceleratorService,
          "PerformanceStatistics" as CFString,
          kCFAllocatorDefault,
          0
        )?.takeRetainedValue() as? [String: Any],
        let agxTelemetry = agxTelemetryFromPerformanceStatistics(performanceStatistics)
      else {
        continue
      }
      return agxTelemetry
    }
    return nil
  }
}
