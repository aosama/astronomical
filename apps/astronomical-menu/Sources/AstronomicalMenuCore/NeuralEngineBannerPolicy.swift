// Decides when the menu app should raise a macOS banner about the Neural Engine. Kept apart from
// the notification API so every rule here runs in a test without any permission prompt.

import Foundation

struct NeuralEngineBanner: Equatable {
  let title: String
  let body: String
}

struct NeuralEngineBannerPolicy {
  // A set that loads in a few seconds should not raise a banner saying to expect another one, which
  // would then arrive on top of it.
  static let preparationAnnouncementDelay: TimeInterval = 5

  private var preparationStartedAt: Date?
  private var preparingModelIdentifier: String?
  private var hasAnnouncedPreparation = false
  private var lastEngagedModelIdentifier: String?
  private var hasObservedDaemon = false

  /// The banner this status calls for, if any. Call with every status the app receives.
  mutating func banner(for statusDocument: DaemonStatusDocument, at now: Date)
    -> NeuralEngineBanner?
  {
    // A missed poll says nothing about the programs, so it must not restart the wait or lose it.
    guard statusDocument.status != "unavailable" else { return nil }
    defer { hasObservedDaemon = true }
    // During a swap the daemon still names the previous model, which is already gone.
    guard !statusDocument.isLoadingModel else {
      forgetPreparation()
      return nil
    }
    let modelIdentifier = statusDocument.readyModelIdentifier
    guard let neuralEnginePrefill = statusDocument.neuralEnginePrefill else {
      forgetPreparation()
      return nil
    }
    if neuralEnginePrefill.isPreparing {
      return bannerWhilePreparing(modelIdentifier: modelIdentifier, at: now)
    }
    if neuralEnginePrefill.isEngaged {
      return bannerWhenEngaged(modelIdentifier: modelIdentifier)
    }
    if neuralEnginePrefill.isUnavailable {
      return bannerWhenUnavailable(modelIdentifier: modelIdentifier)
    }
    forgetPreparation()
    return nil
  }

  private mutating func bannerWhilePreparing(modelIdentifier: String?, at now: Date)
    -> NeuralEngineBanner?
  {
    if preparationStartedAt == nil || preparingModelIdentifier != modelIdentifier {
      preparationStartedAt = now
      preparingModelIdentifier = modelIdentifier
      hasAnnouncedPreparation = false
    }
    guard !hasAnnouncedPreparation, let preparationStartedAt,
      now.timeIntervalSince(preparationStartedAt) >= Self.preparationAnnouncementDelay
    else { return nil }
    hasAnnouncedPreparation = true
    return NeuralEngineBanner(
      title: "Prepping Apple Neural Engine",
      body: "Will notify you when engaged. \(Self.modelTitle(modelIdentifier)) is on the GPU for now.")
  }

  private mutating func bannerWhenEngaged(modelIdentifier: String?) -> NeuralEngineBanner? {
    // A set too quick for a poll to catch still deserves its banner, but a model that was already
    // engaged when this app started has no news in it.
    let hasNews =
      preparationStartedAt != nil
      || (hasObservedDaemon && lastEngagedModelIdentifier != modelIdentifier)
    forgetPreparation()
    lastEngagedModelIdentifier = modelIdentifier
    guard hasNews else { return nil }
    return NeuralEngineBanner(
      title: "Apple Neural Engine ready and engaged",
      body: "\(Self.modelTitle(modelIdentifier)) now prefills on the Neural Engine and the GPU.")
  }

  private mutating func bannerWhenUnavailable(modelIdentifier: String?) -> NeuralEngineBanner? {
    // Only a preparation that promised a result owes one. A model refused at load never did.
    guard preparationStartedAt != nil else { return nil }
    forgetPreparation()
    return NeuralEngineBanner(
      title: "Apple Neural Engine unavailable",
      body: "\(Self.modelTitle(modelIdentifier)) stays on the GPU. The menu shows why.")
  }

  private mutating func forgetPreparation() {
    preparationStartedAt = nil
    preparingModelIdentifier = nil
    hasAnnouncedPreparation = false
  }

  private static func modelTitle(_ modelIdentifier: String?) -> String {
    modelIdentifier ?? "The model"
  }
}
