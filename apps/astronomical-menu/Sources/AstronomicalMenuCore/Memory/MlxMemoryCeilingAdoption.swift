import Foundation

/// Bytes in one gigabyte, matching the daemon and the memory stepper.
let mlxMemoryBytesPerGigabyte: UInt64 = 1_073_741_824

/// Share of a wired claim kept out of MLX planning, in parts per thousand.
///
/// This is the same 25 / 1_000 margin as `plannable_wired_residency_bytes`
/// in the serving ceiling. The menu has to repeat it: the worker publishes
/// the clamped ceiling, and the success sentence must clear when that clamped
/// ceiling arrives rather than waiting for the raw request.
private let metalWorkingSetMarginPartsPerThousand: UInt64 = 25

/// Reports whether the live engine ceiling is the ceiling this request installs.
///
/// A request above the plannable wired budget lands below the number the
/// stepper sent. Treating only an exact byte match as success leaves
/// "MLX memory setting persisted and applied" on screen after the engine has
/// already adopted the clamped ceiling.
func liveMlxMemoryCeilingAdoptedRequest(
  liveCeilingBytes: UInt64,
  requestedCeilingBytes: UInt64,
  wiredResidencyBudgetBytes: UInt64?
) -> Bool {
  liveCeilingBytes == mlxMemoryCeilingTheEngineWillInstall(
    requestedCeilingBytes: requestedCeilingBytes,
    wiredResidencyBudgetBytes: wiredResidencyBudgetBytes
  )
}

/// Ceiling the worker installs for a request: the request itself, or the
/// plannable wired residency when that budget is smaller.
func mlxMemoryCeilingTheEngineWillInstall(
  requestedCeilingBytes: UInt64,
  wiredResidencyBudgetBytes: UInt64?
) -> UInt64 {
  guard let wiredResidencyBudgetBytes else {
    return requestedCeilingBytes
  }
  let plannableWiredResidencyBytes = plannableWiredResidencyBytes(
    claimedWiredResidencyBudgetBytes: wiredResidencyBudgetBytes)
  if plannableWiredResidencyBytes < requestedCeilingBytes {
    return plannableWiredResidencyBytes
  }
  return requestedCeilingBytes
}

/// Wired residency the engine may fill: the claim minus the Metal working-set margin.
private func plannableWiredResidencyBytes(claimedWiredResidencyBudgetBytes: UInt64) -> UInt64 {
  let metalWorkingSetMarginBytes =
    claimedWiredResidencyBudgetBytes / 1_000 * metalWorkingSetMarginPartsPerThousand
  if metalWorkingSetMarginBytes > claimedWiredResidencyBudgetBytes {
    return 0
  }
  return claimedWiredResidencyBudgetBytes - metalWorkingSetMarginBytes
}
