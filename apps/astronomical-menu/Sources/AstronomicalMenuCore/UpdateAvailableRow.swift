import SwiftUI

/// The popover's line about a newer version, shown until the app is updated.
///
/// The button opens the main app's updater, which shows what is new and does
/// the installing; this row never starts a download itself. The state is
/// carried by the words and the arrow, not by colour, so it reads the same in
/// light and dark and does not compete with the telemetry below it.
struct UpdateAvailableRow: View {
  let availableUpdate: AvailableUpdate
  let openUpdater: () -> Void

  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: "arrow.up.circle")
        .accessibilityHidden(true)
      Text(Self.title(for: availableUpdate))
        .lineLimit(1)
      Spacer()
      Button("Update…", action: openUpdater)
        .controlSize(.small)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    .accessibilityElement(children: .contain)
    .accessibilityLabel(Self.title(for: availableUpdate))
  }

  /// Plain text with no view state behind it, so it is callable from any
  /// context; a view's members are otherwise tied to the main actor.
  nonisolated static func title(for availableUpdate: AvailableUpdate) -> String {
    "Version \(availableUpdate.displayVersion) is available"
  }
}
