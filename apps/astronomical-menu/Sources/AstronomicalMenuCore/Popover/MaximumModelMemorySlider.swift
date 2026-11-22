// The popover's "Maximum model RAM" control: a slider from the smallest limit the engine accepts
// to the largest it serves unchanged, with a mark at the loaded model's size and a sentence that
// says what the chosen limit means. It replaced a stepper that gave the number and no meaning.

import SwiftUI

private enum MaximumModelMemorySliderLayout {
  static let thumbDiameter: CGFloat = 18
  static let trackHeight: CGFloat = 6
  static let controlHeight: CGFloat = 22
}

struct MaximumModelMemorySlider: View {
  @ObservedObject var telemetryStore: TelemetryStore
  let statusDocument: DaemonStatusDocument
  /// The Mac's own memory, so the sentence can say what the limit leaves for everything else.
  var macMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
  @State private var isDragging = false

  /// A change is waiting for the reply being written to finish. The engine holds one such change
  /// at a time and refuses a second, so the slider rests until it lands rather than letting a
  /// drag end in a refusal.
  private var changeIsQueued: Bool { statusDocument.pendingMlxMemoryCeilingBytes != nil }

  private var sliderPresentation: MaximumModelMemorySliderPresentation {
    MaximumModelMemorySliderPresentation(
      minimumWholeGigabytes: telemetryStore.minimumWholeGigabytes,
      maximumWholeGigabytes: telemetryStore.maximumWholeGigabytes,
      chosenWholeGigabytes: telemetryStore.editableMaximumMlxMemoryGigabytes,
      modelSizeBytes: statusDocument.isLoadingModel ? nil : statusDocument.readyModelSizeBytes,
      macMemoryBytes: macMemoryBytes)
  }

  var body: some View {
    if telemetryStore.minimumWholeGigabytes <= telemetryStore.maximumWholeGigabytes,
      telemetryStore.maximumWholeGigabytes > 0
    {
      let sliderPresentation = sliderPresentation
      VStack(alignment: .leading, spacing: 4) {
        HStack(alignment: .firstTextBaseline) {
          Text("Maximum model RAM").foregroundStyle(.secondary)
          Spacer()
          Text("\(sliderPresentation.chosenWholeGigabytes) GB")
            .font(PopoverTypography.monospacedBody)
        }
        sliderTrack(sliderPresentation)
        rangeCaptions(sliderPresentation)
        Text(sliderPresentation.explanation)
          .font(.caption)
          // Reading part of a model from the SSD is how Astronomical runs models larger than
          // the limit, so the sentence stays in the quiet colour: it informs, it does not warn.
          .foregroundStyle(.secondary)
          // Two lines are always reserved: the sentence grows from one line to two as the
          // thumb crosses the model's size, and the rows below must not jump under a drag.
          .lineLimit(2, reservesSpace: true)
          // Without a fixed vertical size the popover squeezed the sentence onto one line
          // and cut it off, although the second line was reserved.
          .fixedSize(horizontal: false, vertical: true)
          .frame(maxWidth: .infinity, alignment: .leading)
        followUpRows
      }
    }
  }

  private func sliderTrack(_ sliderPresentation: MaximumModelMemorySliderPresentation) -> some View {
    GeometryReader { geometry in
      let thumbTravel = max(
        geometry.size.width - MaximumModelMemorySliderLayout.thumbDiameter, 1)
      let thumbRadius = MaximumModelMemorySliderLayout.thumbDiameter / 2
      let thumbCenter = thumbRadius + thumbTravel * sliderPresentation.chosenFraction
      ZStack(alignment: .leading) {
        Capsule().fill(.secondary.opacity(0.18))
          .frame(height: MaximumModelMemorySliderLayout.trackHeight)
        Capsule().fill(AstronomicalBrandPalette.accentColor)
          .frame(width: thumbCenter, height: MaximumModelMemorySliderLayout.trackHeight)
        if let modelMarkFraction = sliderPresentation.modelMarkFraction {
          Capsule().fill(Color.primary.opacity(0.7))
            .frame(width: 2, height: MaximumModelMemorySliderLayout.controlHeight - 6)
            .offset(x: thumbRadius + thumbTravel * modelMarkFraction - 1)
        }
        Circle().fill(Color.white)
          .overlay(Circle().stroke(AstronomicalBrandPalette.accentColor, lineWidth: 2))
          .frame(
            width: MaximumModelMemorySliderLayout.thumbDiameter,
            height: MaximumModelMemorySliderLayout.thumbDiameter
          )
          .scaleEffect(isDragging ? 1.12 : 1)
          .offset(x: thumbCenter - thumbRadius)
      }
      .frame(height: MaximumModelMemorySliderLayout.controlHeight)
      .contentShape(Rectangle())
      .gesture(dragGesture(thumbRadius: thumbRadius, thumbTravel: thumbTravel))
    }
    .frame(height: MaximumModelMemorySliderLayout.controlHeight)
    .opacity(changeIsQueued ? 0.45 : 1)
    .allowsHitTesting(!changeIsQueued)
    .accessibilityElement()
    .accessibilityLabel("Maximum model RAM")
    .accessibilityValue("\(sliderPresentation.chosenWholeGigabytes) gigabytes")
    .accessibilityAdjustableAction { adjustmentDirection in
      stepForAccessibility(adjustmentDirection)
    }
  }

  // The limit is applied when the drag ends, never while it moves: applying one can make the
  // engine shed memory, and a drag passes through every value on the way to the one wanted.
  private func dragGesture(thumbRadius: CGFloat, thumbTravel: CGFloat) -> some Gesture {
    DragGesture(minimumDistance: 0)
      .onChanged { dragValue in
        isDragging = true
        telemetryStore.editableMaximumMlxMemoryGigabytes = sliderPresentation.wholeGigabytes(
          atFraction: Double((dragValue.location.x - thumbRadius) / thumbTravel))
      }
      .onEnded { _ in
        isDragging = false
        applyChosenLimitIfChanged()
      }
  }

  // A VoiceOver step moves the thumb without applying: reaching a value takes many steps, and
  // applying each one would have the engine resize its memory once per gigabyte on the way. The
  // Apply button, which appears as soon as the thumb differs from the limit in force, commits.
  private func stepForAccessibility(_ adjustmentDirection: AccessibilityAdjustmentDirection) {
    guard !changeIsQueued else { return }
    let chosenWholeGigabytes = telemetryStore.editableMaximumMlxMemoryGigabytes
    switch adjustmentDirection {
    case .increment where chosenWholeGigabytes < telemetryStore.maximumWholeGigabytes:
      telemetryStore.editableMaximumMlxMemoryGigabytes = chosenWholeGigabytes + 1
    case .decrement where chosenWholeGigabytes > telemetryStore.minimumWholeGigabytes:
      telemetryStore.editableMaximumMlxMemoryGigabytes = chosenWholeGigabytes - 1
    default:
      return
    }
  }

  private func applyChosenLimitIfChanged() {
    switch maximumModelMemoryCommit(
      chosenWholeGigabytes: telemetryStore.editableMaximumMlxMemoryGigabytes,
      maximumWholeGigabytes: telemetryStore.maximumWholeGigabytes,
      appliedWholeGigabytes: appliedMaximumMlxMemoryWholeGigabytes(for: statusDocument),
      configuredWholeGigabytes: statusDocument.configuredMaximumMlxMemoryGigabytes)
    {
    case .nothing: return
    case .followTheMacLimit: telemetryStore.restoreMacMaximumMlxMemoryLimit()
    case .setLimit: telemetryStore.updateMaximumMlxMemoryLimit()
    }
  }

  private func rangeCaptions(_ sliderPresentation: MaximumModelMemorySliderPresentation) -> some View {
    HStack(spacing: 4) {
      Text("\(sliderPresentation.minimumWholeGigabytes) GB")
      Spacer()
      if let modelMarkCaption = sliderPresentation.modelMarkCaption {
        Capsule().fill(Color.primary.opacity(0.7)).frame(width: 2, height: 9)
        Text(modelMarkCaption)
        Spacer()
      }
      // "Max" marks the top as this Mac's own ceiling, which is what resting there selects.
      Text("Max \(sliderPresentation.maximumWholeGigabytes) GB")
    }
    .font(.caption2)
    .foregroundStyle(.secondary)
    .accessibilityHidden(true)
  }

  /// Rows that appear only when there is something to act on or to know, so the control stays
  /// quiet while the engine serves what was asked for.
  @ViewBuilder
  private var followUpRows: some View {
    let controlPresentation = telemetryStore.maximumMlxMemoryControlPresentation
    let updateIsInProgress = telemetryStore.controlActionFeedback?.isInProgress == true
    if !isDragging, !updateIsInProgress, controlPresentation.shouldOfferApply {
      HStack {
        Spacer()
        // Apply remains for the cases a drag does not cover: a thumb moved by VoiceOver, a limit
        // in config.json above what this Mac can serve, and a change that failed and can be
        // tried again.
        Button("Apply", action: applyChosenLimitIfChanged).controlSize(.small)
      }
    }
    if let cappedCeilingCaption = controlPresentation.cappedCeilingCaption {
      Text(cappedCeilingCaption).font(.caption).foregroundStyle(.orange)
    }
    if let pendingMlxMemoryCeilingBytes = statusDocument.pendingMlxMemoryCeilingBytes {
      // A queued change is the control working as designed, so it reads as a quiet note.
      Text(
        queuedMaximumModelMemoryCaption(
          pendingCeilingBytes: pendingMlxMemoryCeilingBytes,
          wiredResidencyBudgetBytes: statusDocument.wiredResidencyBudgetBytes)
      )
      .font(.caption).foregroundStyle(.secondary)
    }
    if let mlxMemoryLimitError = statusDocument.mlxMemoryLimitError {
      Text(mlxMemoryLimitError).font(.caption).foregroundStyle(.red)
    }
  }
}
