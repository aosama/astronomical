// Draws the popover's two speed tiles; SessionSpeedTilesPresentation decides what they say and
// TelemetryStore refreshes them once a second.

import SwiftUI

struct SessionSpeedTiles: View {
  let tilesPresentation: SessionSpeedTilesPresentation

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack(alignment: .top, spacing: 16) {
        tileView(tilesPresentation.readingTile)
        tileView(tilesPresentation.writingTile)
      }
      Text(tilesPresentation.caption).font(.caption2).foregroundStyle(.secondary)
    }
  }

  private func tileView(_ speedTile: SessionSpeedTile) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(speedTile.title).font(.caption).foregroundStyle(.secondary)
      Text(speedTile.speedTitle).font(PopoverTypography.monospacedBody).lineLimit(1)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .help(speedTile.explanation)
    .accessibilityElement(children: .combine)
    .accessibilityLabel(
      "\(speedTile.title): \(speedTile.speedTitle). \(speedTile.explanation)"
    )
  }
}
