import Foundation

/// Largest whole-GB maximum the engine installs unchanged on this Mac.
///
/// This floors the ceiling the engine would install for a machine-maximum
/// request — the machine ceiling, or the plannable wired budget when macOS
/// grants less — rather than the raw wired claim. Flooring the raw claim let
/// the stepper offer a top value the engine then cut by its 2.5% Metal
/// margin, so the panel needed a standing warning to explain the gap.
func reachableMaximumMlxMemoryWholeGigabytes(for statusDocument: DaemonStatusDocument) -> UInt64 {
  mlxMemoryCeilingTheEngineWillInstall(
    requestedCeilingBytes: statusDocument.machineMlxMemoryCeilingBytes,
    wiredResidencyBudgetBytes: statusDocument.wiredResidencyBudgetBytes
  ) / mlxMemoryBytesPerGigabyte
}

/// Whole-GB value the stepper rests on while nobody edits it: the configured
/// maximum, or the reachable maximum while the Mac maximum is in force. It
/// never exceeds the reachable maximum, so the stepper value stays inside its
/// range even when the configuration file asks for more.
func appliedMaximumMlxMemoryWholeGigabytes(for statusDocument: DaemonStatusDocument) -> UInt64 {
  let reachableWholeGigabytes = reachableMaximumMlxMemoryWholeGigabytes(for: statusDocument)
  guard let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes else {
    return reachableWholeGigabytes
  }
  return min(configuredWholeGigabytes, reachableWholeGigabytes)
}

/// What the "Maximum model RAM" control shows.
///
/// The panel is quiet when the engine serves what was asked for: one control
/// row and one grey caption naming the single limit that matters. Buttons
/// appear only when pressing them would change something, and colour appears
/// only when the live ceiling falls short of an explicit request, so orange
/// keeps meaning "look at this".
struct MaximumMlxMemoryControlPresentation: Equatable {
  let reachableMaximumCaption: String
  let shouldOfferApply: Bool
  let shouldOfferMacMaximum: Bool
  let cappedCeilingCaption: String?

  init(statusDocument: DaemonStatusDocument, editableMaximumWholeGigabytes: UInt64) {
    let reachableWholeGigabytes = reachableMaximumMlxMemoryWholeGigabytes(for: statusDocument)
    reachableMaximumCaption = "Up to \(reachableWholeGigabytes) GB on this Mac"
    // A configuration above the reachable maximum also earns Apply: the
    // stepper already rests on the reachable value, so one click replaces the
    // request the engine can never serve and clears the cap notice.
    var configurationIsUnreachable = false
    if let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes {
      configurationIsUnreachable = configuredWholeGigabytes > reachableWholeGigabytes
    }
    shouldOfferApply =
      editableMaximumWholeGigabytes != appliedMaximumMlxMemoryWholeGigabytes(for: statusDocument)
      || configurationIsUnreachable
    shouldOfferMacMaximum = statusDocument.configuredMaximumMlxMemoryGigabytes != nil
    cappedCeilingCaption = Self.cappedCeilingCaption(for: statusDocument)
  }

  /// A cap is reported only against an explicit request: "Use Max" asks for
  /// whatever the Mac allows, so landing below the machine ceiling there is
  /// the request being honoured, not a shortfall. A queued change is skipped
  /// too, because the live ceiling is still the previous one.
  private static func cappedCeilingCaption(for statusDocument: DaemonStatusDocument) -> String? {
    guard let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes,
      statusDocument.pendingMlxMemoryCeilingBytes == nil,
      statusDocument.mlxMemoryCeilingBytes > 0
    else { return nil }
    let (configuredCeilingBytes, overflowOccurred) =
      configuredWholeGigabytes.multipliedReportingOverflow(by: mlxMemoryBytesPerGigabyte)
    guard overflowOccurred || configuredCeilingBytes > statusDocument.mlxMemoryCeilingBytes
    else { return nil }
    return "Capped at \(gigabyteValueText(byteCount: statusDocument.mlxMemoryCeilingBytes)) on this Mac"
  }
}
