// Draws the popover's two speed tiles; SessionSpeedTilesPresentation decides what they say.

import SwiftUI

struct SessionSpeedTiles: View {
  let statusDocument: DaemonStatusDocument

  var body: some View {
    let tilesPresentation = SessionSpeedTilesPresentation(statusDocument: statusDocument)
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
      HStack(spacing: 4) {
        Text(speedTile.title).font(.caption).foregroundStyle(.secondary)
        if speedTile.isLive {
          Text("now")
            .font(.caption2)
            .padding(.horizontal, 5)
            .background(AstronomicalBrandPalette.accentColor.opacity(0.2), in: Capsule())
        }
      }
      Text(speedTile.speedTitle).font(PopoverTypography.monospacedBody).lineLimit(1)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .help(speedTile.explanation)
    .accessibilityElement(children: .combine)
    .accessibilityLabel(
      "\(speedTile.title)\(speedTile.isLive ? ", now" : ""): \(speedTile.speedTitle). \(speedTile.explanation)"
    )
  }
}
