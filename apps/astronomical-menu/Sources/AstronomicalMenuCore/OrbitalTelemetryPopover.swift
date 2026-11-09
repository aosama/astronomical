import AppKit
import SwiftUI

enum PopoverTypography {
  static let bodyPointSize = NSFont.preferredFont(forTextStyle: .body).pointSize + 1
  static let captionPointSize = NSFont.preferredFont(forTextStyle: .caption1).pointSize + 1
  static let headlinePointSize = NSFont.preferredFont(forTextStyle: .headline).pointSize + 1

  static let body = Font.system(size: bodyPointSize)
  static let monospacedBody = Font.system(size: bodyPointSize, design: .monospaced)
  static let boldCaption = Font.system(size: captionPointSize, weight: .bold)
  static let semiboldCaption = Font.system(size: captionPointSize, weight: .semibold)
  static let headline = Font.system(size: headlinePointSize, weight: .semibold)
}

struct OrbitalTelemetryPopover: View {
  @Environment(\.colorScheme) private var colorScheme
  @ObservedObject var telemetryStore: TelemetryStore
  let applicationIdentity: ApplicationIdentity
  let openObservatory: () -> Void
  let openDownloadModels: () -> Void
  let openChat: () -> Void
  let reloadConfiguration: () -> Void
  let restartServer: () -> Void
  let checkForUpdates: () -> Void
  @Binding var automaticallyChecksForUpdates: Bool
  @Binding var selectedUpdateChannel: ApplicationUpdateChannel
  let updatesSupported: Bool
  let terminalCommandSupported: Bool
  let installTerminalCommand: () -> Void
  let removeTerminalCommand: () -> Void
  let launchAtLoginSupported: Bool
  @Binding var launchAtLoginIsEnabled: Bool
  let revealConfiguration: () -> Void
  let showWelcome: () -> Void
  let quitApplication: () -> Void

  var body: some View {
    let statusDocument = telemetryStore.statusDocument
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .top) {
        VStack(alignment: .leading, spacing: 4) {
          HStack(spacing: 6) {
            Text("ASTRONOMICAL").font(PopoverTypography.boldCaption).tracking(2).foregroundStyle(
              applicationIdentity.channel == .development ? .orange : .cyan)
            Text(applicationIdentity.channel.displayName.uppercased())
              .font(PopoverTypography.semiboldCaption)
              .padding(.horizontal, 6)
              .padding(.vertical, 2)
              .background(
                applicationIdentity.channel == .development
                  ? .orange.opacity(0.2) : .cyan.opacity(0.2),
                in: Capsule())
          }
          HStack(alignment: .center, spacing: 6) {
            if let modelLoadInFlight = statusDocument.modelLoadInFlight {
              Text(modelLoadInFlight.headlineTitle)
                .font(PopoverTypography.headline)
                .lineLimit(1)
            } else if let readyModelIdentifier = statusDocument.readyModelIdentifier {
              Text(readyModelIdentifier)
                .font(PopoverTypography.headline)
                .lineLimit(1)
            } else {
              Text("No model resident")
                .font(PopoverTypography.headline)
                .lineLimit(1)
            }
          }
        }
        Spacer()
        Text(statusDocument.phaseTitle)
          .font(PopoverTypography.semiboldCaption)
          .padding(.horizontal, 8).padding(.vertical, 4)
          .background(
            statusDocument.isActive || statusDocument.isLoadingModel
              ? .cyan.opacity(0.2) : .secondary.opacity(0.15), in: Capsule())
      }
      HStack(alignment: .top, spacing: 16) {
        summaryMetric("Size on disk", statusDocument.modelDiskSizeTitle)
        residencySummary(statusDocument)
      }
      if statusDocument.status == "unavailable",
        let statusRefreshErrorMessage = telemetryStore.lastStatusRefreshErrorMessage
      {
        StatusRefreshErrorView(message: statusRefreshErrorMessage)
      }
      Divider()
      metricRow("Flight", statusDocument.flightTitle)
      PopoverProgressRow(statusDocument: statusDocument)
      metricRow(statusDocument.elapsedTimeMetricTitle, statusDocument.elapsedTimeTitle)
      if let neuralEngineTitle = statusDocument.neuralEnginePrefill?.metricTitle(
        showsEstimate: !statusDocument.isActive)
      {
        metricRow("Neural Engine", neuralEngineTitle)
        if let neuralEngineReason = statusDocument.neuralEnginePrefill?.unavailableReasonTitle {
          Text(neuralEngineReason)
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
      Divider()
      GPUUtilizationBar(
        gpuUtilizationPercentage: telemetryStore.systemTelemetrySnapshot.gpuUtilizationPercentage
      )
      MemoryPressureIndicator(
        memoryPressureTitle: telemetryStore.systemTelemetrySnapshot.memoryPressureTitle
      )
      MlxMemoryBreakdownBar(
        activeByteCount: statusDocument.mlxMemoryActiveBytes,
        limitByteCount: statusDocument.mlxMemoryCeilingBytes,
        breakdown: statusDocument.mlxMemoryBreakdown,
        headroomSplit: statusDocument.mlxHeadroomSplit,
        sourceTitle: statusDocument.mlxMemorySourceTitle
      )
      maximumMlxMemoryControl(statusDocument)
      metricRow("Session", statusDocument.sessionTitle)
      PromptReuseBar(
        reusedFraction: statusDocument.sessionPromptReuseFraction,
        percentageTitle: statusDocument.sessionPromptReusePercentageTitle,
        breakdownTitle: statusDocument.sessionPromptReuseBreakdownTitle
      )
      if let controlActionFeedback = telemetryStore.controlActionFeedback {
        Text(controlActionFeedback.message)
          .font(.caption)
          .foregroundStyle(
            controlActionFeedback.isFailure ? .red : controlActionFeedback.isInProgress ? .secondary : .green
          )
          .accessibilityLabel("Server control: \(controlActionFeedback.message)")
      }
      if let discoveryDiagnostic = telemetryStore.statusDocument.configuration?
        .modelDiscoveryDiagnostics?.first
      {
        VStack(alignment: .leading, spacing: 6) {
          Text(discoveryDiagnostic.message)
            .font(.caption)
            .foregroundStyle(.orange)
          Button("Reveal config", action: revealConfiguration)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Model discovery warning. \(discoveryDiagnostic.message)")
      }
      Divider()
      VStack(alignment: .leading, spacing: 2) {
        if let serverApplication = telemetryStore.statusDocument.application {
          Text(serverApplication.buildTitle)
            .font(.caption.monospaced())
        } else {
          Text(applicationIdentity.buildTitle)
            .font(.caption.monospaced())
        }
        if let serverIdentity = telemetryStore.statusDocument.application,
          serverIdentity.version != applicationIdentity.version
            || serverIdentity.commit != applicationIdentity.commit
        {
          Text("App bundle: \(applicationIdentity.buildTitle)")
            .font(.caption)
            .foregroundStyle(.orange)
        }
        Text("Config: \(applicationIdentity.expectedConfigurationFile) · Port \(applicationIdentity.daemonPort)")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
      Button(action: telemetryStore.hasDiscoveredModels ? openObservatory : openDownloadModels) {
        Label(
          telemetryStore.hasDiscoveredModels ? "Open Observatory" : "Download a model",
          systemImage: telemetryStore.hasDiscoveredModels ? "safari" : "arrow.down.circle"
        )
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .tint(.cyan)
      Button(action: openChat) {
        Label("Chat", systemImage: "bubble.left.and.bubble.right")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .tint(.indigo)
      HStack {
        Button("Reload config", action: reloadConfiguration)
        Button("Restart server", action: restartServer)
        Spacer()
        Menu {
          if updatesSupported {
            Button("Check for Updates…", action: checkForUpdates)
            Toggle("Check automatically", isOn: $automaticallyChecksForUpdates)
            Picker("Update channel", selection: $selectedUpdateChannel) {
              ForEach(ApplicationUpdateChannel.allCases) { updateChannel in
                Text(updateChannel.displayName).tag(updateChannel)
              }
            }
            Divider()
          }
          if terminalCommandSupported {
            Button("Install “astronomical” command in terminal…", action: installTerminalCommand)
            Button("Remove “astronomical” terminal command…", action: removeTerminalCommand)
            Divider()
          }
          if launchAtLoginSupported {
            Toggle("Launch at Login", isOn: $launchAtLoginIsEnabled)
            Divider()
          }
          Button("Welcome…", action: showWelcome)
          Button("Reveal config", action: revealConfiguration)
          Button("Quit Astronomical", action: quitApplication)
        } label: {
          Image(systemName: "ellipsis.circle")
        }
        .menuStyle(.borderlessButton)
      }
    }
    .font(PopoverTypography.body)
    .padding(16).frame(width: 390).background(
      orbitalTelemetryBackgroundColor(colorScheme: colorScheme)
    )
  }

  private func metricRow(_ metricLabel: String, _ metricText: String) -> some View {
    HStack {
      Text(metricLabel).foregroundStyle(.secondary)
      Spacer()
      Text(metricText).font(PopoverTypography.monospacedBody)
    }
  }

  private func summaryMetric(_ metricLabel: String, _ metricText: String) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(metricLabel).font(.caption).foregroundStyle(.secondary)
      Text(metricText).font(PopoverTypography.monospacedBody).lineLimit(1)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func residencySummary(_ statusDocument: DaemonStatusDocument) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text("Residency").font(.caption).foregroundStyle(.secondary)
      // The daemon keeps these labels short enough for one line of this
      // half-width cell; the scale factor only covers an unexpected label
      // (or a larger system font), so the row never wraps.
      // Width arithmetic: the popover is 390pt wide; minus 32pt padding and
      // 16pt column spacing leaves about 171pt per cell. The monospaced body
      // is the system body size + 1 (about 14pt), about 8.4pt per character,
      // so about 20 characters fit and the daemon keeps labels to 18.
      Text(statusDocument.modelFootprintTitle)
        .font(PopoverTypography.monospacedBody)
        .lineLimit(1)
        .minimumScaleFactor(0.75)
      // The daemon's reply sentence ("Experts are being served from RAM.")
      // is left to Observatory: it wrapped to two more lines here and the
      // popover is already long, while the label above says the same thing.
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  @ViewBuilder
  private func maximumMlxMemoryControl(_ statusDocument: DaemonStatusDocument) -> some View {
    if telemetryStore.minimumWholeGigabytes <= telemetryStore.maximumWholeGigabytes,
      telemetryStore.maximumWholeGigabytes > 0
    {
      let controlPresentation = telemetryStore.maximumMlxMemoryControlPresentation
      VStack(alignment: .leading, spacing: 4) {
        // One row: the popover is long, so the label, the stepper and the
        // reachable maximum share a line instead of stacking three.
        HStack(spacing: 6) {
          // The label is the one piece allowed to shrink, so Apply and Use Max
          // showing together still fit the 358pt row instead of clipping.
          Text("Maximum model RAM").foregroundStyle(.secondary)
            .lineLimit(1).minimumScaleFactor(0.7)
          Spacer(minLength: 4)
          Stepper(
            value: $telemetryStore.editableMaximumMlxMemoryGigabytes,
            in: telemetryStore.minimumWholeGigabytes...telemetryStore.maximumWholeGigabytes
          ) {
            Text("\(telemetryStore.editableMaximumMlxMemoryGigabytes) GB")
              .font(PopoverTypography.monospacedBody)
          }
          .fixedSize()
          Text(controlPresentation.reachableMaximumCaption)
            .font(.caption).foregroundStyle(.secondary).lineLimit(1).fixedSize()
          if controlPresentation.shouldOfferApply {
            Button("Apply", action: telemetryStore.updateMaximumMlxMemoryLimit)
              .controlSize(.small)
          }
          if controlPresentation.shouldOfferMacMaximum {
            Button("Use Max", action: telemetryStore.restoreMacMaximumMlxMemoryLimit)
              .controlSize(.small)
              .accessibilityLabel("Use the Mac maximum")
          }
        }
        if let cappedCeilingCaption = controlPresentation.cappedCeilingCaption {
          Text(cappedCeilingCaption).font(.caption).foregroundStyle(.orange)
        }
        if let pendingMlxMemoryCeilingBytes = statusDocument.pendingMlxMemoryCeilingBytes {
          Text("Pending \(gigabyteValueText(byteCount: pendingMlxMemoryCeilingBytes))")
            .font(.caption).foregroundStyle(.orange)
        }
        if let mlxMemoryLimitError = statusDocument.mlxMemoryLimitError {
          Text(mlxMemoryLimitError).font(.caption).foregroundStyle(.red)
        }
      }
      .accessibilityElement(children: .contain)
    }
  }
}

func orbitalTelemetryBackgroundColor(colorScheme: ColorScheme) -> Color {
  if colorScheme == .dark {
    return Color(.sRGB, red: 24 / 255, green: 25 / 255, blue: 25 / 255, opacity: 1)
  }
  return Color(nsColor: .windowBackgroundColor)
}

struct GPUUtilizationBar: View {
  let gpuUtilizationPercentage: Double?

  // Statements are not allowed in a ViewBuilder block, so the optional-to-
  // zero fallback lives beside the view instead of inside its body.
  private var observedUtilizationFraction: Double {
    if let gpuUtilizationPercentage {
      return gpuUtilizationPercentage / 100
    }
    return 0
  }

  var body: some View {
    VStack(spacing: 5) {
      HStack {
        Text("GPU utilization").foregroundStyle(.secondary)
        Spacer()
        if let gpuUtilizationPercentage {
          Text("\(Int(gpuUtilizationPercentage.rounded()))%")
            .font(PopoverTypography.monospacedBody)
        } else {
          Text("Unavailable")
            .font(PopoverTypography.monospacedBody)
        }
      }
      HorizontalUsageBar(
        usageFraction: usageBarFillFraction(observedUtilizationFraction),
        fillColor: .green
      )
    }
    .accessibilityElement(children: .combine)
  }
}

struct MemoryPressureIndicator: View {
  let memoryPressureTitle: SystemMemoryPressureTitle

  var body: some View {
    HStack {
      Text("Memory pressure").foregroundStyle(.secondary)
      Spacer()
      Text(memoryPressureTitle.rawValue)
        .font(PopoverTypography.monospacedBody)
        .foregroundStyle(memoryPressureTitle.tintColor)
    }
    .accessibilityElement(children: .combine)
    .accessibilityLabel("macOS memory pressure: \(memoryPressureTitle.rawValue)")
  }
}

extension SystemMemoryPressureTitle {
  var tintColor: Color {
    switch self {
    case .normal: return .green
    case .warning: return .orange
    case .critical: return .red
    case .unavailable: return .secondary
    }
  }
}

struct PromptReuseBar: View {
  let reusedFraction: Double
  let percentageTitle: String
  let breakdownTitle: String

  var body: some View {
    VStack(spacing: 5) {
      HStack {
        Text("Cache efficacy").foregroundStyle(.secondary)
        Spacer()
        Text(percentageTitle).font(PopoverTypography.monospacedBody)
      }
      HorizontalUsageBar(usageFraction: usageBarFillFraction(reusedFraction), fillColor: .cyan)
      HStack {
        Text("Target work").foregroundStyle(.secondary)
        Spacer()
        Text(breakdownTitle).font(PopoverTypography.monospacedBody)
      }
    }
    .accessibilityElement(children: .combine)
  }
}

/// Fill fraction for a usage bar's track. Out-of-range and NaN inputs clamp to
/// the track so a fill can never render outside its capsule.
func usageBarFillFraction(_ fraction: Double) -> Double {
  if fraction.isNaN { return 0 }
  return min(1, max(0, fraction))
}

struct HorizontalUsageBar: View {
  let usageFraction: Double
  let fillColor: Color

  var body: some View {
    GeometryReader { geometry in
      ZStack(alignment: .leading) {
        Capsule().fill(.secondary.opacity(0.18))
        Capsule().fill(fillColor).frame(width: geometry.size.width * usageBarFillFraction(usageFraction))
      }
    }
    .frame(height: 7)
  }
}
