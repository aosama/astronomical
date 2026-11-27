// What the "Maximum model RAM" slider shows for one position of its thumb: where the thumb and
// the model-size mark sit on the track, and one sentence saying what the chosen limit means.

import Foundation

/// What letting go of the slider asks the engine to do.
enum MaximumModelMemoryCommit: Equatable {
  case nothing
  /// Clear the setting, so the limit follows whatever this Mac allows.
  case followTheMacLimit
  case setLimit(wholeGigabytes: UInt64)
}

/// The top of the slider means "this Mac's limit", not the number printed there: it clears the
/// setting instead of saving that number, so the limit keeps following the Mac if what the Mac
/// allows changes later. That is why the control needs no separate reset button.
func maximumModelMemoryCommit(
  chosenWholeGigabytes: UInt64,
  maximumWholeGigabytes: UInt64,
  appliedWholeGigabytes: UInt64,
  configuredWholeGigabytes: UInt64?
) -> MaximumModelMemoryCommit {
  if chosenWholeGigabytes >= maximumWholeGigabytes {
    return configuredWholeGigabytes == nil ? .nothing : .followTheMacLimit
  }
  return chosenWholeGigabytes == appliedWholeGigabytes
    ? .nothing : .setLimit(wholeGigabytes: chosenWholeGigabytes)
}

struct MaximumModelMemorySliderPresentation: Equatable {
  let minimumWholeGigabytes: UInt64
  let maximumWholeGigabytes: UInt64
  let chosenWholeGigabytes: UInt64
  /// Size on disk of the loaded model; `nil` when none is loaded.
  let modelSizeBytes: UInt64?
  let macMemoryBytes: UInt64

  private var rangeWholeGigabytes: Double {
    Double(maximumWholeGigabytes.saturatingSubtracting(minimumWholeGigabytes))
  }

  private func trackFraction(forGigabytes gigabytes: Double) -> Double {
    guard rangeWholeGigabytes > 0 else { return 0 }
    return min(max((gigabytes - Double(minimumWholeGigabytes)) / rangeWholeGigabytes, 0), 1)
  }

  var chosenFraction: Double { trackFraction(forGigabytes: Double(chosenWholeGigabytes)) }

  /// The limit under a point on the track, snapped to a whole gigabyte because the engine and
  /// config.json store whole gigabytes.
  func wholeGigabytes(atFraction trackFraction: Double) -> UInt64 {
    let clampedFraction = min(max(trackFraction, 0), 1)
    return minimumWholeGigabytes + UInt64((clampedFraction * rangeWholeGigabytes).rounded())
  }

  /// Where the loaded model's size falls on the track. No mark is drawn when the size lies
  /// outside the range: every position is then on the same side of it, and the sentence says so.
  var modelMarkFraction: Double? {
    guard let modelSizeBytes, rangeWholeGigabytes > 0 else { return nil }
    let modelSizeGigabytes = Double(modelSizeBytes) / Double(mlxMemoryBytesPerGigabyte)
    guard modelSizeGigabytes > Double(minimumWholeGigabytes),
      modelSizeGigabytes < Double(maximumWholeGigabytes)
    else { return nil }
    return trackFraction(forGigabytes: modelSizeGigabytes)
  }

  var modelMarkCaption: String? {
    guard modelMarkFraction != nil, let modelSizeBytes else { return nil }
    return "Model size \(gigabyteValueText(byteCount: modelSizeBytes))"
  }

  /// A limit below the model's own size cannot hold all of it, so part is read from the SSD.
  /// Only that certain case is called out: above the size the model still needs working room,
  /// so "fits" is left to the Residency row, which reports what the engine actually did.
  var partOfTheModelIsReadFromTheSSD: Bool {
    guard let modelSizeBytes else { return false }
    let (chosenBytes, overflowOccurred) =
      chosenWholeGigabytes.multipliedReportingOverflow(by: mlxMemoryBytesPerGigabyte)
    return !overflowOccurred && chosenBytes < modelSizeBytes
  }

  var explanation: String {
    let macMemoryWholeGigabytes = UInt64(
      (Double(macMemoryBytes) / Double(mlxMemoryBytesPerGigabyte)).rounded())
    let remainingWholeGigabytes = macMemoryWholeGigabytes.saturatingSubtracting(chosenWholeGigabytes)
    guard partOfTheModelIsReadFromTheSSD, let modelSizeBytes else {
      return "Leaves \(remainingWholeGigabytes) GB for macOS and your other apps."
    }
    // Short enough for the two lines the popover gives it: the earlier, longer sentence was cut
    // off before it reached what the limit leaves for other apps.
    return
      "At \(chosenWholeGigabytes) GB, part of this \(gigabyteValueText(byteCount: modelSizeBytes)) model is read from the SSD, which is slower. Leaves \(remainingWholeGigabytes) GB for other apps."
  }
}
