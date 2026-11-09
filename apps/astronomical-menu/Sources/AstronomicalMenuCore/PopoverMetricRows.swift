import SwiftUI

/// One label-and-value line of the popover's telemetry section.
struct PopoverMetricRow: View {
  let metricLabel: String
  let metricText: String

  var body: some View {
    HStack {
      Text(metricLabel).foregroundStyle(.secondary)
      Spacer()
      Text(metricText).font(PopoverTypography.monospacedBody)
    }
  }
}

/// The Progress line, with the progress bar between the label and the value.
///
/// The bar shares the line rather than taking one of its own, so it can appear
/// and disappear between phases without moving the rows below it or leaving an
/// empty band where it would be. There is no spinner: the Flight and Progress
/// rows already say what is happening.
struct PopoverProgressRow: View {
  let statusDocument: DaemonStatusDocument

  var body: some View {
    HStack(spacing: 10) {
      // The label never wraps or shrinks; only the bar gives up width.
      Text("Progress").foregroundStyle(.secondary).lineLimit(1).fixedSize()
      if statusDocument.showsProgressBar {
        // The popover's capsule rather than a `ProgressView`: the system
        // linear bar is taller than a line of body text, so it would grow
        // the row whenever it appeared.
        HorizontalUsageBar(usageFraction: statusDocument.progressBarFillFraction, fillColor: .cyan)
          .frame(minWidth: 40)
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
    .accessibilityElement(children: .combine)
    .accessibilityValue(statusDocument.showsProgressBar ? statusDocument.progressBarAccessibilityValue : "")
  }
}
