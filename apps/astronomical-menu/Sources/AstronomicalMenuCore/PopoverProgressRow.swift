import SwiftUI

/// The popover's Progress row, with the progress bar between the label and
/// the value.
///
/// The bar used to sit on its own line between the Flight and Progress rows.
/// That line kept its height while the bar was hidden, so the rows below would
/// not jump between phases, and it left an empty band whenever there was no
/// bar (a generation without an output cap, a model load). Inside the row the
/// bar adds no height, so it can come and go without moving anything. There is
/// no spinner: the Flight and Progress rows already say what is happening.
struct PopoverProgressRow: View {
  let statusDocument: DaemonStatusDocument

  var body: some View {
    HStack(spacing: 10) {
      Text("Progress").foregroundStyle(.secondary)
      if statusDocument.showsProgressBar {
        // The popover's own 7pt capsule rather than a `ProgressView`: the
        // system linear bar is 20pt tall, taller than this 17pt text line, so
        // it would grow the row by 3pt whenever it appeared. The bar is
        // hidden from VoiceOver because the value beside it already reads out
        // the same progress.
        HorizontalUsageBar(
          usageFraction: usageBarFillFraction(
            Double(statusDocument.progressCompletedUnitCount)
              / Double(max(statusDocument.progressTotalUnitCount, 1))),
          fillColor: .cyan
        )
        .frame(minWidth: 40)
        .accessibilityHidden(true)
      } else {
        Spacer()
      }
      // The value outranks the bar for width, so a long title such as
      // "45% · 1234 / 10000 tokens" narrows the bar rather than wrapping.
      Text(statusDocument.progressTitle)
        .font(PopoverTypography.monospacedBody)
        .lineLimit(1)
        .layoutPriority(1)
    }
  }
}
