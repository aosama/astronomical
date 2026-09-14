import SwiftUI

/// The single-screen first-run welcome shown by the menu application
/// (issue #610): the user can tell the local runner is live on this Mac and
/// sees exactly one next step. The content observes the same live telemetry
/// store as the popover, so the running claim is truthful rather than a static
/// splash, and the one call to action follows the popover's existing branch:
/// Library when no model is discovered, Observatory when one is.
struct FirstRunWelcomeView: View {
  @ObservedObject var telemetryStore: TelemetryStore
  let applicationIdentity: ApplicationIdentity
  let openObservatory: () -> Void
  let openLibrary: () -> Void
  let restartServer: () -> Void
  let dismissWelcome: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      HStack(spacing: 6) {
        Image(systemName: "sparkles").foregroundStyle(.cyan)
        Text("ASTRONOMICAL").font(PopoverTypography.boldCaption).tracking(2).foregroundStyle(.cyan)
        Text(applicationIdentity.channel.displayName.uppercased())
          .font(PopoverTypography.semiboldCaption)
          .padding(.horizontal, 6).padding(.vertical, 2)
          .background(
            applicationIdentity.channel == .development
              ? .orange.opacity(0.2) : .cyan.opacity(0.2),
            in: Capsule())
      }

      VStack(alignment: .leading, spacing: 6) {
        Text(runningHeadline).font(.title2.weight(.semibold))
        Text("Your models and conversations stay on this Mac.")
          .font(PopoverTypography.body).foregroundStyle(.secondary)
      }

      if statusDocument.status == "unavailable" {
        HStack {
          Text("The local runner is not reachable yet.").foregroundStyle(.secondary)
          Spacer()
          Button("Restart", action: restartServer).controlSize(.small)
        }
        .padding(10)
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
      }

      VStack(alignment: .leading, spacing: 4) {
        Text("Model").font(.caption).foregroundStyle(.secondary)
        Text(statusDocument.readyModelIdentifier ?? "No model resident yet")
          .font(PopoverTypography.monospacedBody).lineLimit(1)
      }

      Button(action: primaryAction) {
        Label(primaryActionTitle, systemImage: primaryActionSymbol)
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .tint(.cyan)
      .controlSize(.large)

      HStack {
        Spacer()
        Button("I'll explore on my own", action: dismissWelcome)
          .buttonStyle(.plain)
          .foregroundStyle(.secondary)
        Spacer()
      }
    }
    .padding(24)
    .frame(width: 460)
  }

  private var statusDocument: SupervisorStatusDocument {
    telemetryStore.statusDocument
  }

  // The headline is the "it is running on this Mac" outcome, phrased from the
  // live status document instead of an assumption made at launch.
  private var runningHeadline: String {
    switch statusDocument.status {
    case "ready", "loading": return "Running locally on this Mac"
    case "unavailable": return "Starting up on this Mac"
    default: return "Running locally on this Mac"
    }
  }

  private var primaryActionTitle: String {
    telemetryStore.hasDiscoveredModels ? "Open Observatory" : "Download a model"
  }

  private var primaryActionSymbol: String {
    telemetryStore.hasDiscoveredModels ? "safari" : "arrow.down.circle"
  }

  private var primaryAction: () -> Void {
    telemetryStore.hasDiscoveredModels ? openObservatory : openLibrary
  }
}
