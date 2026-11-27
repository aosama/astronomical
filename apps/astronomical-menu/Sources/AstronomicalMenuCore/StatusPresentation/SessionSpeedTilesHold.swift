// When the popover's speed tiles change: at most once a second, a timing choice kept apart from
// what the tiles say (SessionSpeedTilesPresentation).

/// Shows the tiles at most once a second. The status is polled four times a second while a request
/// runs, for the progress bar; figures that changed four times a second would be hard to read.
/// The first figures, and the figures when the popover opens, show at once.
struct SessionSpeedTilesHold {
  static let refreshInterval = Duration.seconds(1)

  private(set) var shownPresentation: SessionSpeedTilesPresentation?
  private var shownAt: ContinuousClock.Instant?

  /// The tiles to show at `now` for the latest status: the latest figures when a second has passed
  /// since the shown ones (or nothing is shown yet, or `showsAtOnce`), otherwise the shown ones.
  mutating func presentation(
    for latestPresentation: SessionSpeedTilesPresentation, at now: ContinuousClock.Instant,
    showsAtOnce: Bool = false
  ) -> SessionSpeedTilesPresentation {
    let isRefreshDue =
      shownAt.map { $0.duration(to: now) >= Self.refreshInterval } ?? true
    if let shownPresentation, !showsAtOnce, !isRefreshDue {
      return shownPresentation
    }
    shownPresentation = latestPresentation
    shownAt = now
    return latestPresentation
  }
}
