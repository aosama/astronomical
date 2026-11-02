// Draws the Night plate mark as the menu bar glyph and decides which state it
// shows. The glyph uses the product's state texture (solid = in memory,
// hatched = streaming from the SSD) so the menu bar answers "where does the
// model live right now" without opening the popover.

import AppKit

enum MenuBarGlyphState: Equatable {
  /// No model is ready: an empty plate.
  case idle
  /// The ready model's experts are held in memory.
  case resident
  /// The ready model pages experts from the SSD.
  case streaming
  /// A request is live; the reticle fills, and the status title beside the
  /// glyph says what is running, so the state never depends on colour.
  case generating

  var accessibilityTitle: String {
    switch self {
    case .idle: return "idle"
    case .resident: return "model in memory"
    case .streaming: return "streaming experts from SSD"
    case .generating: return "generating"
    }
  }
}

extension DaemonStatusDocument {
  var menuBarGlyphState: MenuBarGlyphState {
    guard status == "ready", readyModelIdentifier != nil else { return .idle }
    if isActive { return .generating }
    // "paged" and "hybrid" both stream some experts from the SSD; dense
    // models report no mode and are wholly in memory.
    switch expertMemoryMode {
    case "paged", "hybrid": return .streaming
    default: return .resident
    }
  }
}

private let menuBarGlyphPointSize: CGFloat = 18

@MainActor
func makeMenuBarGlyphImage(state: MenuBarGlyphState) -> NSImage {
  let glyphImage = NSImage(
    size: NSSize(width: menuBarGlyphPointSize, height: menuBarGlyphPointSize),
    flipped: true
  ) { glyphBounds in
    drawMenuBarGlyph(state: state, in: glyphBounds)
    return true
  }
  // Template rendering lets AppKit tint the glyph for light, dark and
  // highlighted menu bars, so only alpha carries meaning here.
  glyphImage.isTemplate = true
  glyphImage.accessibilityDescription = "Astronomical, \(state.accessibilityTitle)"
  return glyphImage
}

// Coordinates follow site/assets/mark.svg on a 64-unit plate (top-left origin),
// scaled into the plate rectangle; the drawing context is flipped to match.
private struct MenuBarPlate {
  let plateRectangle: NSRect

  func point(markX: CGFloat, markY: CGFloat) -> NSPoint {
    NSPoint(
      x: plateRectangle.minX + markX * plateRectangle.width / 64,
      y: plateRectangle.minY + markY * plateRectangle.height / 64)
  }

  func circle(markX: CGFloat, markY: CGFloat, markRadius: CGFloat) -> NSBezierPath {
    let center = point(markX: markX, markY: markY)
    let radius = markRadius * plateRectangle.width / 64
    return NSBezierPath(
      ovalIn: NSRect(
        x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
  }
}

private func drawMenuBarGlyph(state: MenuBarGlyphState, in glyphBounds: NSRect) {
  let plate = MenuBarPlate(plateRectangle: glyphBounds.insetBy(dx: 1.5, dy: 1.5))
  let platePath = NSBezierPath(roundedRect: plate.plateRectangle, xRadius: 3.5, yRadius: 3.5)
  NSColor.black.setStroke()
  NSColor.black.setFill()

  NSGraphicsContext.saveGraphicsState()
  platePath.addClip()
  switch state {
  case .idle:
    break
  case .resident, .generating:
    plate.circle(markX: 8, markY: 92, markRadius: 62).fill()
  case .streaming:
    drawHatchedDisc(plate: plate)
  }
  NSGraphicsContext.restoreGraphicsState()

  platePath.lineWidth = 1.2
  platePath.stroke()

  let reticlePath = plate.circle(markX: 44, markY: 20, markRadius: 9)
  if state == .generating {
    reticlePath.fill()
  } else {
    reticlePath.lineWidth = 1.1
    reticlePath.stroke()
  }
}

private func drawHatchedDisc(plate: MenuBarPlate) {
  let discPath = plate.circle(markX: 8, markY: 92, markRadius: 62)
  discPath.lineWidth = 1.1
  discPath.stroke()

  NSGraphicsContext.saveGraphicsState()
  discPath.addClip()
  // A 2.5 pt period puts one stripe on every few physical pixels at 2x, the
  // coarsest spacing that still reads as texture rather than as a grey fill.
  let stripePeriod: CGFloat = 2.5
  let plateRectangle = plate.plateRectangle
  let stripePath = NSBezierPath()
  stripePath.lineWidth = 1
  var stripeOffset = -plateRectangle.height
  while stripeOffset < plateRectangle.width {
    stripePath.move(to: NSPoint(x: plateRectangle.minX + stripeOffset, y: plateRectangle.maxY))
    stripePath.line(
      to: NSPoint(x: plateRectangle.minX + stripeOffset + plateRectangle.height, y: plateRectangle.minY))
    stripeOffset += stripePeriod
  }
  stripePath.stroke()
  NSGraphicsContext.restoreGraphicsState()
}
