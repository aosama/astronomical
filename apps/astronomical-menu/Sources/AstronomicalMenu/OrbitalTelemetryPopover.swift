import AppKit
import SwiftUI

private enum PopoverTypography {
  static let bodyPointSize = NSFont.preferredFont(forTextStyle: .body).pointSize + 1
  static let captionPointSize = NSFont.preferredFont(forTextStyle: .caption1).pointSize + 1
  static let headlinePointSize = NSFont.preferredFont(forTextStyle: .headline).pointSize + 1

  static let body = Font.system(size: bodyPointSize)
  static let monospacedBody = Font.system(size: bodyPointSize, design: .monospaced)
  static let boldCaption = Font.system(size: captionPointSize, weight: .bold)
  static let semiboldCaption = Font.system(size: captionPointSize, weight: .semibold)
  static let headline = Font.system(size: headlinePointSize, weight: .semibold)
}

enum MlxMemoryPalette {
  static let experts = Color(.sRGB, red: 10 / 255, green: 132 / 255, blue: 255 / 255, opacity: 1)
  static let modelCore = Color(.sRGB, red: 86 / 255, green: 180 / 255, blue: 233 / 255, opacity: 1)
  static let contextState = Color(.sRGB, red: 240 / 255, green: 228 / 255, blue: 66 / 255, opacity: 1)
  static let runtimeWork = Color(.sRGB, red: 167 / 255, green: 139 / 255, blue: 250 / 255, opacity: 1)
  static let available = Color.secondary.opacity(0.18)
  static let segmentDivider = Color(.sRGB, red: 24 / 255, green: 25 / 255, blue: 25 / 255, opacity: 1)
}

enum MlxMemoryLegendItem: Equatable {
  case experts
  case modelCore
  case contextState
  case runtimeWork
  case available

  var title: String {
    switch self {
    case .experts: return "Experts"
    case .modelCore: return "Model core"
    case .contextState: return "Live context state"
    case .runtimeWork: return "Runtime work"
    case .available: return "Nominal MLX headroom"
    }
  }

  var explanationText: String {
    switch self {
    case .experts:
      return "Sparse MoE weights currently resident in MLX, including loaded expert pages."
    case .modelCore:
      return "Always-resident non-expert weights, including embeddings, attention, and vision weights."
    case .contextState:
      return "Decoder state for the active request, including conversation key-value state. It is released after completion and is separate from the client conversation window."
    case .runtimeWork:
      return "Temporary computation work and other active MLX memory not attributed above."
    case .available:
      return "Calculated capacity below this Mac's MLX ceiling. It is not free RAM; macOS memory pressure and temporary work can reduce what is safely usable."
    }
  }

  var infoButtonAccessibilityLabel: String { "Explain \(title)" }
}

struct OrbitalTelemetryPopover: View {
  @Environment(\.colorScheme) private var colorScheme
  @ObservedObject var telemetryStore: TelemetryStore
  let openObservatory: () -> Void
  let reloadConfiguration: () -> Void
  let restartServer: () -> Void
  let revealConfiguration: () -> Void
  let quitApplication: () -> Void

  var body: some View {
    let statusDocument = telemetryStore.statusDocument
    VStack(alignment: .leading, spacing: 14) {
      HStack(alignment: .top) {
        VStack(alignment: .leading, spacing: 4) {
          Text("ASTRONOMICAL").font(PopoverTypography.boldCaption).tracking(2).foregroundStyle(
            .cyan)
          Text(statusDocument.readyModelIdentifier ?? "No model resident").font(
            PopoverTypography.headline
          )
          .lineLimit(1)
        }
        Spacer()
        Text(statusDocument.phaseTitle)
          .font(PopoverTypography.semiboldCaption)
          .padding(.horizontal, 8).padding(.vertical, 4)
          .background(
            statusDocument.isActive ? .cyan.opacity(0.2) : .secondary.opacity(0.15), in: Capsule())
      }
      metricRow("Size on disk", statusDocument.modelDiskSizeTitle)
      metricRow("Generation mode", statusDocument.mtpRuntimeStateTitle)
      if let mtpUnavailableReason = statusDocument.mtpUnavailableReason {
        Text(mtpUnavailableReason)
          .font(.caption)
          .foregroundStyle(.secondary)
          .lineLimit(2)
          .frame(maxWidth: .infinity, alignment: .trailing)
      }
      metricRow("Residency", statusDocument.modelFootprintTitle)
      Divider()
      metricRow("Flight", statusDocument.flightTitle)
      ProgressView(
        value: Double(statusDocument.progressProcessedTokenCount),
        total: Double(statusDocument.progressTotalTokenCount)
      ).tint(.cyan)
      metricRow("Progress", statusDocument.progressTitle)
      metricRow(statusDocument.elapsedTimeMetricTitle, statusDocument.elapsedTimeTitle)
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
      Divider()
      Button(action: openObservatory) {
        Label("Open Observatory", systemImage: "safari")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .tint(.cyan)
      HStack {
        Button("Reload config", action: reloadConfiguration)
        Button("Restart server", action: restartServer)
        Spacer()
        Menu {
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

  @ViewBuilder
  private func maximumMlxMemoryControl(_ statusDocument: SupervisorStatusDocument) -> some View {
    if telemetryStore.minimumWholeDecimalGigabytes <= telemetryStore.maximumWholeDecimalGigabytes,
      telemetryStore.maximumWholeDecimalGigabytes > 0
    {
      VStack(alignment: .leading, spacing: 6) {
        Text("Maximum model RAM").foregroundStyle(.secondary)
        HStack {
          Stepper(
            value: $telemetryStore.editableMaximumMlxMemoryGigabytes,
            in: telemetryStore.minimumWholeDecimalGigabytes...telemetryStore.maximumWholeDecimalGigabytes
          ) {
            Text("\(telemetryStore.editableMaximumMlxMemoryGigabytes) GB")
              .font(PopoverTypography.monospacedBody)
          }
          Button("Apply", action: telemetryStore.updateMaximumMlxMemoryLimit)
          Button("Use Mac Maximum", action: telemetryStore.restoreMacMaximumMlxMemoryLimit)
        }
        Text(
          "Effective \(decimalGigabyteValueText(byteCount: statusDocument.mlxMemoryCeilingBytes)) of \(decimalGigabyteValueText(byteCount: statusDocument.machineMlxMemoryCeilingBytes))"
        ).font(.caption).foregroundStyle(.secondary)
        if let pendingMlxMemoryCeilingBytes = statusDocument.pendingMlxMemoryCeilingBytes {
          Text("Pending \(decimalGigabyteValueText(byteCount: pendingMlxMemoryCeilingBytes))")
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

struct MlxMemoryBreakdownBar: View {
  let activeByteCount: UInt64
  let limitByteCount: UInt64
  let breakdown: SupervisorStatusDocument.MlxMemoryBreakdown
  let sourceTitle: String
  @State private var selectedMemoryLegendItem: MlxMemoryLegendItem?

  var body: some View {
    VStack(spacing: 6) {
      HStack {
        Text("MLX memory").foregroundStyle(.secondary)
        Spacer()
        Text(
          "\(decimalGigabyteText(byteCount: activeByteCount)) / \(decimalGigabyteText(byteCount: limitByteCount))"
        ).font(PopoverTypography.monospacedBody)
      }
      Text(sourceTitle).font(.caption).foregroundStyle(.secondary)
      GeometryReader { geometry in
        HStack(spacing: 0) {
          memorySegment(MlxMemoryPalette.experts, breakdown.expertPayloadByteCount, geometry.size.width)
          memorySegment(MlxMemoryPalette.modelCore, breakdown.modelCorePayloadByteCount, geometry.size.width)
          memorySegment(
            MlxMemoryPalette.runtimeWork,
            breakdown.runtimeWorkByteCount,
            geometry.size.width
          )
          memorySegment(
            MlxMemoryPalette.contextState,
            breakdown.contextStatePayloadByteCount,
            geometry.size.width,
            showsTrailingDivider: breakdown.contextStatePayloadByteCount > 0 && breakdown.availableByteCount > 0
          )
          memorySegment(MlxMemoryPalette.available, breakdown.availableByteCount, geometry.size.width)
        }
        .clipShape(Capsule())
      }
      .frame(height: 7)
      memoryLegendRow(.experts, MlxMemoryPalette.experts, breakdown.expertPayloadByteCount)
      memoryLegendRow(.modelCore, MlxMemoryPalette.modelCore, breakdown.modelCorePayloadByteCount)
      memoryLegendRow(.runtimeWork, MlxMemoryPalette.runtimeWork, breakdown.runtimeWorkByteCount)
      memoryLegendRow(.contextState, MlxMemoryPalette.contextState, breakdown.contextStatePayloadByteCount)
      memoryLegendRow(.available, MlxMemoryPalette.available, breakdown.availableByteCount)
    }
    .accessibilityElement(children: .combine)
    .accessibilityLabel(
      "MLX memory, \(sourceTitle). Latest worker observation; colors are reconciled ownership estimates."
    )
  }

  private func memorySegment(
    _ color: Color,
    _ byteCount: UInt64,
    _ width: CGFloat,
    showsTrailingDivider: Bool = false
  ) -> some View {
    color
      .frame(width: width * memoryBreakdownFraction(byteCount, limitByteCount))
      .overlay(alignment: .trailing) {
        if showsTrailingDivider {
          MlxMemoryPalette.segmentDivider.frame(width: 1)
        }
      }
  }

  private func memoryLegendRow(
    _ legendItem: MlxMemoryLegendItem,
    _ color: Color,
    _ byteCount: UInt64
  ) -> some View {
    HStack {
      RoundedRectangle(cornerRadius: 3).fill(color).frame(width: 10, height: 10)
      Text(legendItem.title).foregroundStyle(.secondary)
      Button {
        selectedMemoryLegendItem = legendItem
      } label: {
        Image(systemName: "info.circle")
      }
      .buttonStyle(.plain)
      .foregroundStyle(.secondary)
      .accessibilityLabel(legendItem.infoButtonAccessibilityLabel)
      .popover(isPresented: memoryLegendPopoverIsPresented(for: legendItem)) {
        Text(legendItem.explanationText)
          .frame(width: 260, alignment: .leading)
          .padding(12)
      }
      Spacer()
      Text(decimalGigabyteValueText(byteCount: byteCount)).font(PopoverTypography.monospacedBody)
    }
  }

  private func memoryLegendPopoverIsPresented(for legendItem: MlxMemoryLegendItem) -> Binding<Bool> {
    Binding(
      get: { selectedMemoryLegendItem == legendItem },
      set: { shouldPresentPopover in
        if shouldPresentPopover {
          selectedMemoryLegendItem = legendItem
        } else if selectedMemoryLegendItem == legendItem {
          selectedMemoryLegendItem = nil
        }
      }
    )
  }
}

func memoryBreakdownFraction(_ byteCount: UInt64, _ limitByteCount: UInt64) -> Double {
  guard limitByteCount > 0 else { return 0 }
  return min(1, Double(byteCount) / Double(limitByteCount))
}

struct GPUUtilizationBar: View {
  let gpuUtilizationPercentage: Double?

  var body: some View {
    VStack(spacing: 5) {
      HStack {
        Text("GPU utilization").foregroundStyle(.secondary)
        Spacer()
        Text(gpuUtilizationPercentage.map { "\(Int($0.rounded()))%" } ?? "Unavailable")
          .font(PopoverTypography.monospacedBody)
      }
      HorizontalUsageBar(
        usageFraction: min(1, max(0, (gpuUtilizationPercentage ?? 0) / 100)),
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
      HorizontalUsageBar(usageFraction: reusedFraction, fillColor: .cyan)
      HStack {
        Text("Target + drafter work").foregroundStyle(.secondary)
        Spacer()
        Text(breakdownTitle).font(PopoverTypography.monospacedBody)
      }
    }
    .accessibilityElement(children: .combine)
  }
}

struct HorizontalUsageBar: View {
  let usageFraction: Double
  let fillColor: Color

  var body: some View {
    GeometryReader { geometry in
      ZStack(alignment: .leading) {
        Capsule().fill(.secondary.opacity(0.18))
        Capsule().fill(fillColor).frame(width: geometry.size.width * usageFraction)
      }
    }
    .frame(height: 7)
  }
}
