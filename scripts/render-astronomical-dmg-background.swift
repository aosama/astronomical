#!/usr/bin/env swift

// Renders the nonessential visual cue; named icons remain the accessible installation fallback.

import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
  FileHandle.standardError.write(Data("Usage: render-astronomical-dmg-background.swift OUTPUT\n".utf8))
  exit(2)
}

let outputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let canvasWidth = 640
let canvasHeight = 388
guard let bitmap = NSBitmapImageRep(
  bitmapDataPlanes: nil,
  pixelsWide: canvasWidth,
  pixelsHigh: canvasHeight,
  bitsPerSample: 8,
  samplesPerPixel: 4,
  hasAlpha: true,
  isPlanar: false,
  colorSpaceName: .deviceRGB,
  bytesPerRow: 0,
  bitsPerPixel: 0
) else {
  FileHandle.standardError.write(Data("Error: could not create DMG background bitmap\n".utf8))
  exit(1)
}
guard let graphicsContext = NSGraphicsContext(bitmapImageRep: bitmap) else {
  FileHandle.standardError.write(Data("Error: could not create DMG background graphics context\n".utf8))
  exit(1)
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = graphicsContext
NSColor(calibratedRed: 0.075, green: 0.082, blue: 0.071, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight).fill()

let centeredParagraph = NSMutableParagraphStyle()
centeredParagraph.alignment = .center
let titleAttributes: [NSAttributedString.Key: Any] = [
  .font: NSFont.systemFont(ofSize: 24, weight: .semibold),
  .foregroundColor: NSColor(calibratedRed: 0.957, green: 0.941, blue: 0.902, alpha: 1),
  .paragraphStyle: centeredParagraph,
]
("Drag Astronomical to Applications" as NSString).draw(
  in: NSRect(x: 70, y: 322, width: 500, height: 40),
  withAttributes: titleAttributes
)

let arrowPath = NSBezierPath()
arrowPath.move(to: NSPoint(x: 270, y: 185))
arrowPath.line(to: NSPoint(x: 370, y: 185))
arrowPath.move(to: NSPoint(x: 340, y: 215))
arrowPath.line(to: NSPoint(x: 370, y: 185))
arrowPath.line(to: NSPoint(x: 340, y: 155))
arrowPath.lineWidth = 8
arrowPath.lineCapStyle = .round
arrowPath.lineJoinStyle = .round
NSColor(calibratedRed: 0.847, green: 1, blue: 0.243, alpha: 0.88).setStroke()
arrowPath.stroke()
NSGraphicsContext.restoreGraphicsState()

guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
  FileHandle.standardError.write(Data("Error: could not encode DMG background\n".utf8))
  exit(1)
}
do {
  try FileManager.default.createDirectory(
    at: outputURL.deletingLastPathComponent(),
    withIntermediateDirectories: true
  )
  try pngData.write(to: outputURL, options: .atomic)
} catch {
  FileHandle.standardError.write(Data("Error: could not write DMG background: \(error)\n".utf8))
  exit(1)
}
print("[dmg-background] status=success pixels=\(canvasWidth)x\(canvasHeight)")
