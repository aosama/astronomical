// The wording and bar length of the popover's one-line memory row (EverydayMemoryRow draws it).

import Foundation

extension DaemonStatusDocument {
  /// "12.40 GB of 36.50 GB" while a model is loaded; the limit alone when nothing is held, so the
  /// row still says how much a model may use.
  var everydayMemoryTitle: String {
    guard mlxMemoryCeilingBytes > 0 else { return "Not measured" }
    let limitTitle = gigabyteValueText(byteCount: mlxMemoryCeilingBytes)
    guard mlxMemoryActiveBytes > 0 else { return "None used of \(limitTitle)" }
    return "\(gigabyteValueText(byteCount: mlxMemoryActiveBytes)) of \(limitTitle)"
  }

  var everydayMemoryFillFraction: Double {
    guard mlxMemoryCeilingBytes > 0 else { return 0 }
    return usageBarFillFraction(Double(mlxMemoryActiveBytes) / Double(mlxMemoryCeilingBytes))
  }
}
