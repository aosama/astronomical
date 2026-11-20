// Draws the Astronomical mark as the menu bar glyph.

import AppKit

private let menuBarGlyphPointSize: CGFloat = 18

@MainActor
func makeMenuBarGlyphImage(_ glyph: MenuBarGlyph) -> NSImage {
  let glyphImage = NSImage(
    size: NSSize(width: menuBarGlyphPointSize, height: menuBarGlyphPointSize),
    flipped: true
  ) { glyphBounds in
    drawMenuBarGlyph(glyph, in: glyphBounds)
    return true
  }
  // Template rendering lets AppKit tint the glyph for light, dark and
  // highlighted menu bars, so only alpha carries meaning here.
  glyphImage.isTemplate = true
  glyphImage.accessibilityDescription = "Astronomical, \(glyph.accessibilityTitle)"
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

  func length(markLength: CGFloat) -> CGFloat { markLength * plateRectangle.width / 64 }

  func circle(markX: CGFloat, markY: CGFloat, markRadius: CGFloat) -> NSBezierPath {
    let center = point(markX: markX, markY: markY)
    let radius = length(markLength: markRadius)
    return NSBezierPath(
      ovalIn: NSRect(
        x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
  }
}

private func drawMenuBarGlyph(_ glyph: MenuBarGlyph, in glyphBounds: NSRect) {
  let plate = MenuBarPlate(plateRectangle: glyphBounds.insetBy(dx: 1.5, dy: 1.5))
  let platePath = NSBezierPath(roundedRect: plate.plateRectangle, xRadius: 3.5, yRadius: 3.5)
  NSColor.black.setStroke()
  NSColor.black.setFill()

  NSGraphicsContext.saveGraphicsState()
  platePath.addClip()
  switch glyph.state {
  case .idle, .engineNotRunning:
    break
  case .resident, .generating:
    plate.circle(markX: 8, markY: 92, markRadius: 62).fill()
  case .streaming, .loadingModel:
    drawHatchedDisc(plate: plate)
  }
  NSGraphicsContext.restoreGraphicsState()

  platePath.lineWidth = 1.2
  platePath.stroke()

  if glyph.state == .engineNotRunning {
    drawExclamationMark(plate: plate)
  } else if let completedProgressStepCount = glyph.completedProgressStepCount {
    drawProgressPie(plate: plate, completedProgressStepCount: completedProgressStepCount)
  } else {
    drawReticle(plate: plate, isFilled: glyph.state == .generating || glyph.state == .loadingModel)
  }
}

private func drawReticle(plate: MenuBarPlate, isFilled: Bool) {
  let reticlePath = plate.circle(markX: 44, markY: 20, markRadius: 9)
  if isFilled {
    reticlePath.fill()
  } else {
    reticlePath.lineWidth = 1.1
    reticlePath.stroke()
  }
}

// The reticle grows into a pie while work with a measured end runs. // of 15 units is the largest that stays clear of the disc (18.5 units away)
// and of the plate's edge, and it is the smallest at which a quarter and a
// half are told apart at menu bar size.
private func drawProgressPie(plate: MenuBarPlate, completedProgressStepCount: Int) {
  let pieCenter = plate.point(markX: 44, markY: 20)
  let outlinePath = plate.circle(markX: 44, markY: 20, markRadius: 13)
  outlinePath.lineWidth = 1
  outlinePath.stroke()
  guard completedProgressStepCount > 0 else { return }
  let completedDegrees =
    360 * CGFloat(min(completedProgressStepCount, menuBarGlyphProgressStepCount))
    / CGFloat(menuBarGlyphProgressStepCount)
  let wedgePath = NSBezierPath()
  wedgePath.move(to: pieCenter)
  // The context is flipped, so increasing angles from -90 sweep clockwise from twelve o'clock.
  wedgePath.appendArc(
    withCenter: pieCenter, radius: plate.length(markLength: 13), startAngle: -90,
    endAngle: -90 + completedDegrees, clockwise: false)
  wedgePath.close()
  wedgePath.fill()
}

private func drawExclamationMark(plate: MenuBarPlate) {
  let stemPath = NSBezierPath()
  stemPath.lineWidth = 1.8
  stemPath.lineCapStyle = .round
  stemPath.move(to: plate.point(markX: 32, markY: 15))
  stemPath.line(to: plate.point(markX: 32, markY: 36))
  stemPath.stroke()
  plate.circle(markX: 32, markY: 48, markRadius: 4.5).fill()
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
