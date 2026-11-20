// The popover's one memory line for people who leave technical details off: how much memory the
// model holds against the limit they set, with a single bar. The five-part breakdown answers a
// developer's question ("which part of the model uses it") and stays behind the switch.

import SwiftUI

struct EverydayMemoryRow: View {
  let statusDocument: DaemonStatusDocument

  var body: some View {
    VStack(spacing: 5) {
      HStack {
        Text("Memory").foregroundStyle(.secondary)
        Spacer()
        Text(statusDocument.everydayMemoryTitle).font(PopoverTypography.monospacedBody)
      }
      HorizontalUsageBar(
        usageFraction: statusDocument.everydayMemoryFillFraction,
        fillColor: AstronomicalBrandPalette.accentColor)
    }
    .accessibilityElement(children: .combine)
  }
}
