import AppKit
import SwiftUI

enum MlxMemoryPalette {
  static let experts = Color(.sRGB, red: 10 / 255, green: 132 / 255, blue: 255 / 255, opacity: 1)
  static let modelCore = Color(.sRGB, red: 86 / 255, green: 180 / 255, blue: 233 / 255, opacity: 1)
  static let drafter = Color(.sRGB, red: 230 / 255, green: 159 / 255, blue: 0, opacity: 1)
  static let contextState = Color(.sRGB, red: 240 / 255, green: 228 / 255, blue: 66 / 255, opacity: 1)
  static let runtimeWork = Color(.sRGB, red: 167 / 255, green: 139 / 255, blue: 250 / 255, opacity: 1)
  static let available = Color.secondary.opacity(0.18)
  /// Unused expert budget — Okabe-Ito bluish green, not occupancy.
  static let unusedBudget = Color(
    .sRGB, red: 0 / 255, green: 158 / 255, blue: 115 / 255, opacity: 1)
  /// Context-growth reserve — Okabe-Ito reddish purple, distinct from live context yellow.
  static let reservedContextGrowth = Color(
    .sRGB, red: 204 / 255, green: 121 / 255, blue: 167 / 255, opacity: 1)
  /// Activation and stream-slot reserve — Okabe-Ito vermillion, not the drafter orange.
  static let reservedActivations = Color(
    .sRGB, red: 213 / 255, green: 94 / 255, blue: 0 / 255, opacity: 1)
  static let unexplainedHeadroom = Color.primary.opacity(0.28)
  static let ownerOverrun = Color.red
  static let segmentDivider = Color(.sRGB, red: 24 / 255, green: 25 / 255, blue: 25 / 255, opacity: 1)
}

enum MlxMemoryLegendItem: Equatable {
  case experts
  case modelCore
  case drafter
  case contextState
  case runtimeWork
  case available
  case unusedBudget
  case reservedContextGrowth
  case reservedActivations
  case reservedModelCoreSlack
  case unexplainedHeadroom
  case ownerOverrun

  var title: String {
    switch self {
    case .experts: return "Experts"
    case .modelCore: return "Model core"
    case .drafter: return "Drafter"
    case .contextState: return "Live context state"
    case .runtimeWork: return "Runtime work"
    case .available: return "Nominal MLX headroom"
    case .unusedBudget: return "Unused"
    case .reservedContextGrowth: return "Held for context"
    case .reservedActivations: return "Held for activations"
    case .reservedModelCoreSlack: return "Core slack"
    case .unexplainedHeadroom: return "Unexplained headroom"
    case .ownerOverrun: return "Owner overrun"
    }
  }

  var explanationText: String {
    switch self {
    case .experts:
      return "Sparse MoE weights currently resident in MLX, including loaded expert pages."
    case .modelCore:
      return "Always-resident non-expert weights, including embeddings, attention, and vision weights."
    case .drafter:
      return "All active MLX memory attributed to live request-scoped drafter scoring, including draft weights, resident draft expert pages, temporary decoder context, visual inputs, scoring tensors, and draft-phase work. It returns to zero after the drafter is released and is separate from the client conversation window."
    case .contextState:
      return "Decoder state for the active request, including conversation key-value state. It is released after completion and is separate from the client conversation window."
    case .runtimeWork:
      return "Temporary computation work and other active MLX memory not attributed above."
    case .available:
      return "Calculated capacity below this Mac's MLX ceiling. It is not free RAM; macOS memory pressure and temporary work can reduce what is safely usable."
    case .unusedBudget:
      return "Unused budget the ceiling already granted for expert weights that are not in RAM. Distinct from Experts, which are weights already resident."
    case .reservedContextGrowth:
      return "Held so conversation state can grow. Spending it on expert weights invites eviction mid-request."
    case .reservedActivations:
      return "Held for temporary decode and prefill workspace, including one expert page. It is a promise, not idle RAM."
    case .reservedModelCoreSlack:
      return "Model-core reserve the loaded core does not occupy."
    case .unexplainedHeadroom:
      return "Unused capacity with no named owner yet. The engine reports this instead of folding it into a reserved bucket."
    case .ownerOverrun:
      return "A named owner consumed more than its reserve. This is not unused capacity."
    }
  }

  var infoButtonAccessibilityLabel: String { "Explain \(title)" }
}

struct MlxMemoryBreakdownBar: View {
  let activeByteCount: UInt64
  let limitByteCount: UInt64
  let breakdown: SupervisorStatusDocument.MlxMemoryBreakdown
  let headroomSplit: MlxHeadroomSplit
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
            MlxMemoryPalette.drafter,
            breakdown.speculativePrefillDraftMemoryByteCount,
            geometry.size.width
          )
          memorySegment(
            MlxMemoryPalette.runtimeWork,
            breakdown.runtimeWorkByteCount,
            geometry.size.width
          )
          memorySegment(
            MlxMemoryPalette.contextState,
            breakdown.contextStatePayloadByteCount,
            geometry.size.width,
            showsTrailingDivider: breakdown.contextStatePayloadByteCount > 0
              && breakdown.availableByteCount > 0
          )
          if headroomSplit.enginePublishedTheSplit {
            memorySegment(
              MlxMemoryPalette.unusedBudget,
              headroomSplit.unusedBudgetByteCount,
              geometry.size.width
            )
            memorySegment(
              MlxMemoryPalette.reservedContextGrowth,
              headroomSplit.reservedContextGrowthByteCount,
              geometry.size.width
            )
            memorySegment(
              MlxMemoryPalette.reservedActivations,
              headroomSplit.reservedActivationByteCount,
              geometry.size.width
            )
            memorySegment(
              MlxMemoryPalette.modelCore.opacity(0.45),
              headroomSplit.reservedModelCoreSlackByteCount,
              geometry.size.width
            )
            memorySegment(
              MlxMemoryPalette.unexplainedHeadroom,
              headroomSplit.unexplainedHeadroomByteCount,
              geometry.size.width
            )
            memorySegment(
              MlxMemoryPalette.available,
              headroomSplit.remainderByteCount,
              geometry.size.width
            )
          } else {
            memorySegment(MlxMemoryPalette.available, breakdown.availableByteCount, geometry.size.width)
          }
        }
        .clipShape(Capsule())
      }
      .frame(height: 7)
      memoryLegendRow(.experts, MlxMemoryPalette.experts, breakdown.expertPayloadByteCount)
      memoryLegendRow(.modelCore, MlxMemoryPalette.modelCore, breakdown.modelCorePayloadByteCount)
      memoryLegendRow(
        .drafter,
        MlxMemoryPalette.drafter,
        breakdown.speculativePrefillDraftMemoryByteCount
      )
      memoryLegendRow(.runtimeWork, MlxMemoryPalette.runtimeWork, breakdown.runtimeWorkByteCount)
      memoryLegendRow(.contextState, MlxMemoryPalette.contextState, breakdown.contextStatePayloadByteCount)
      if headroomSplit.enginePublishedTheSplit {
        memoryLegendRow(
          .unusedBudget,
          MlxMemoryPalette.unusedBudget,
          headroomSplit.unusedBudgetByteCount
        )
        memoryLegendRow(
          .reservedContextGrowth,
          MlxMemoryPalette.reservedContextGrowth,
          headroomSplit.reservedContextGrowthByteCount
        )
        memoryLegendRow(
          .reservedActivations,
          MlxMemoryPalette.reservedActivations,
          headroomSplit.reservedActivationByteCount
        )
        if headroomSplit.reservedModelCoreSlackByteCount > 0 {
          memoryLegendRow(
            .reservedModelCoreSlack,
            MlxMemoryPalette.modelCore.opacity(0.45),
            headroomSplit.reservedModelCoreSlackByteCount
          )
        }
        if headroomSplit.unexplainedHeadroomByteCount > 0 {
          memoryLegendRow(
            .unexplainedHeadroom,
            MlxMemoryPalette.unexplainedHeadroom,
            headroomSplit.unexplainedHeadroomByteCount
          )
        }
        if headroomSplit.ownerOverrunByteCount > 0 {
          memoryLegendRow(
            .ownerOverrun,
            MlxMemoryPalette.ownerOverrun,
            headroomSplit.ownerOverrunByteCount
          )
        }
      } else {
        memoryLegendRow(.available, MlxMemoryPalette.available, breakdown.availableByteCount)
      }
    }
    .accessibilityElement(children: .combine)
    .accessibilityLabel(accessibilitySummary)
  }

  private var accessibilitySummary: String {
    if headroomSplit.enginePublishedTheSplit {
      return "MLX memory, \(sourceTitle). Occupied ownership plus named unused-capacity owners from the engine budget."
    }
    return "MLX memory, \(sourceTitle). Latest worker observation; colors are reconciled ownership estimates."
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
