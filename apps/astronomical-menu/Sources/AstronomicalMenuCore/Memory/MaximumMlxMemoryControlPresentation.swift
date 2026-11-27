import Foundation

/// Largest whole-GB maximum the engine installs unchanged on this Mac.
///
/// This floors the ceiling the engine would install for a machine-maximum
/// request — the machine ceiling, or the plannable wired budget when macOS
/// grants less — rather than the raw wired claim. Flooring the raw claim let
/// the control offer a top value the engine then cut by its 2.5% Metal
/// margin, so the panel needed a standing warning to explain the gap.
func reachableMaximumMlxMemoryWholeGigabytes(for statusDocument: DaemonStatusDocument) -> UInt64 {
  mlxMemoryCeilingTheEngineWillInstall(
    requestedCeilingBytes: statusDocument.machineMlxMemoryCeilingBytes,
    wiredResidencyBudgetBytes: statusDocument.wiredResidencyBudgetBytes
  ) / mlxMemoryBytesPerGigabyte
}

/// Whole-GB value the slider rests on while nobody edits it: the configured
/// maximum, or the reachable maximum while the Mac maximum is in force. It
/// never exceeds the reachable maximum, so the slider's thumb stays inside its
/// range even when the configuration file asks for more.
func appliedMaximumMlxMemoryWholeGigabytes(for statusDocument: DaemonStatusDocument) -> UInt64 {
  let reachableWholeGigabytes = reachableMaximumMlxMemoryWholeGigabytes(for: statusDocument)
  guard let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes else {
    return reachableWholeGigabytes
  }
  return min(configuredWholeGigabytes, reachableWholeGigabytes)
}

/// Where the slider's thumb rests after a status poll.
///
/// While the popover is closed the thumb always takes the limit in force. While it is open the
/// thumb may be mid-edit, so it is left alone only when the person has moved it: a thumb still
/// resting on the limit that was in force follows that limit when something else changes it
/// (the Observatory, the command line, a REST request). Leaving it behind showed 18 GB with an
/// Apply button under a memory row that already read "of 34.00 GB".
func maximumModelMemoryThumbAfterStatusPoll(
  chosenWholeGigabytes: UInt64,
  previouslyAppliedWholeGigabytes: UInt64,
  appliedWholeGigabytes: UInt64,
  popoverIsVisible: Bool
) -> UInt64 {
  let thumbRestsOnTheLimitThatWasInForce = chosenWholeGigabytes == previouslyAppliedWholeGigabytes
  if !popoverIsVisible || thumbRestsOnTheLimitThatWasInForce {
    return appliedWholeGigabytes
  }
  return chosenWholeGigabytes
}

/// What the "Maximum model RAM" control offers beside its slider
/// (MaximumModelMemorySlider draws it). The panel is quiet when the engine
/// serves what was asked for. Apply appears only when pressing it would
/// change something, and a cap notice appears only when the live ceiling
/// falls short of an explicit request, so orange keeps meaning "look at this".
struct MaximumMlxMemoryControlPresentation: Equatable {
  let shouldOfferApply: Bool
  let cappedCeilingCaption: String?

  init(statusDocument: DaemonStatusDocument, editableMaximumWholeGigabytes: UInt64) {
    let reachableWholeGigabytes = reachableMaximumMlxMemoryWholeGigabytes(for: statusDocument)
    // A configuration above the reachable maximum also earns Apply: the
    // slider already rests on the reachable value, so one click replaces the
    // request the engine can never serve and clears the cap notice.
    var configurationIsUnreachable = false
    if let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes {
      configurationIsUnreachable = configuredWholeGigabytes > reachableWholeGigabytes
    }
    shouldOfferApply =
      editableMaximumWholeGigabytes != appliedMaximumMlxMemoryWholeGigabytes(for: statusDocument)
      || configurationIsUnreachable
    cappedCeilingCaption = Self.cappedCeilingCaption(for: statusDocument)
  }

  /// A cap is reported only against an explicit request: the top of the slider asks for
  /// whatever the Mac allows, so landing below the machine ceiling there is
  /// the request being honoured, not a shortfall.
  private static func cappedCeilingCaption(for statusDocument: DaemonStatusDocument) -> String? {
    guard let configuredWholeGigabytes = statusDocument.configuredMaximumMlxMemoryGigabytes,
      statusDocument.mlxMemoryCeilingBytes > 0
    else { return nil }
    let (configuredCeilingBytes, overflowOccurred) =
      configuredWholeGigabytes.multipliedReportingOverflow(by: mlxMemoryBytesPerGigabyte)
    guard overflowOccurred || configuredCeilingBytes > statusDocument.mlxMemoryCeilingBytes
    else { return nil }
    return "Capped at \(gigabyteValueText(byteCount: statusDocument.mlxMemoryCeilingBytes)) on this Mac"
  }
}

/// Whether the slider shows, in red, that the engine refused the limit it was last asked for.
/// The engine answers a change at once, even while a reply is being written, so a refusal is
/// known the moment the thumb is let go; red says so before the caption under it is read.
func maximumModelMemoryLimitWasRefused(statusDocument: DaemonStatusDocument) -> Bool {
  statusDocument.mlxMemoryLimitError != nil
}

