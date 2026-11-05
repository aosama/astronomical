#!/usr/bin/env swift

// Renders the nebula mark for Stable and a hatched-disc Development variant.

import AppKit
import Foundation

private struct IconRenderArguments {
  let outputDirectory: URL
  let applicationChannel: ApplicationChannel
}

private enum ApplicationChannel: String {
  case stable
  case development
}

private struct IconVariant {
  let fileName: String
  let pixelSize: Int
}

private enum IconRenderError: LocalizedError {
  case missingArgument(String)
  case duplicateArgument(String)
  case unrecognizedArgument(String)
  case invalidChannel(String)
  case bitmapCreationFailed(Int)
  case graphicsContextCreationFailed(Int)
  case pngEncodingFailed(Int)

  var errorDescription: String? {
    switch self {
    case .missingArgument(let argumentName):
      return "missing required argument: \(argumentName)"
    case .duplicateArgument(let argumentName):
      return "argument may be provided only once: \(argumentName)"
    case .unrecognizedArgument(let argument):
      return "unrecognized argument: \(argument)"
    case .invalidChannel(let channel):
      return "channel must be stable or development, received: \(channel)"
    case .bitmapCreationFailed(let pixelSize):
      return "could not allocate the \(pixelSize)-pixel icon bitmap"
    case .graphicsContextCreationFailed(let pixelSize):
      return "could not create the \(pixelSize)-pixel icon graphics context"
    case .pngEncodingFailed(let pixelSize):
      return "could not encode the \(pixelSize)-pixel icon as PNG"
    }
  }
}

private let iconVariants = [
  IconVariant(fileName: "icon_16x16.png", pixelSize: 16),
  IconVariant(fileName: "icon_16x16@2x.png", pixelSize: 32),
  IconVariant(fileName: "icon_32x32.png", pixelSize: 32),
  IconVariant(fileName: "icon_32x32@2x.png", pixelSize: 64),
  IconVariant(fileName: "icon_128x128.png", pixelSize: 128),
  IconVariant(fileName: "icon_128x128@2x.png", pixelSize: 256),
  IconVariant(fileName: "icon_256x256.png", pixelSize: 256),
  IconVariant(fileName: "icon_256x256@2x.png", pixelSize: 512),
  IconVariant(fileName: "icon_512x512.png", pixelSize: 512),
  IconVariant(fileName: "icon_512x512@2x.png", pixelSize: 1024),
]

private func parseArguments() throws -> IconRenderArguments {
  let commandArguments = Array(CommandLine.arguments.dropFirst())
  var outputDirectoryPath: String?
  var applicationChannel: ApplicationChannel?
  var argumentIndex = 0

  while argumentIndex < commandArguments.count {
    let argumentName = commandArguments[argumentIndex]
    guard argumentIndex + 1 < commandArguments.count else {
      throw IconRenderError.missingArgument(argumentName)
    }
    let argumentValue = commandArguments[argumentIndex + 1]
    switch argumentName {
    case "--output-directory":
      guard outputDirectoryPath == nil else {
        throw IconRenderError.duplicateArgument(argumentName)
      }
      outputDirectoryPath = argumentValue
    case "--channel":
      guard applicationChannel == nil else {
        throw IconRenderError.duplicateArgument(argumentName)
      }
      guard let parsedChannel = ApplicationChannel(rawValue: argumentValue) else {
        throw IconRenderError.invalidChannel(argumentValue)
      }
      applicationChannel = parsedChannel
    default:
      throw IconRenderError.unrecognizedArgument(argumentName)
    }
    argumentIndex += 2
  }

  guard let outputDirectoryPath else {
    throw IconRenderError.missingArgument("--output-directory")
  }
  guard let applicationChannel else {
    throw IconRenderError.missingArgument("--channel")
  }

  return IconRenderArguments(
    outputDirectory: URL(fileURLWithPath: outputDirectoryPath, isDirectory: true),
    applicationChannel: applicationChannel
  )
}

private func renderIcon(
  pixelSize: Int,
  applicationChannel: ApplicationChannel
) throws -> Data {
  guard let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: pixelSize,
    pixelsHigh: pixelSize,
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
  ) else {
    throw IconRenderError.bitmapCreationFailed(pixelSize)
  }
  guard let graphicsContext = NSGraphicsContext(bitmapImageRep: bitmap) else {
    throw IconRenderError.graphicsContextCreationFailed(pixelSize)
  }

  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = graphicsContext
  graphicsContext.imageInterpolation = .high
  drawAstronomicalMark(canvasSize: CGFloat(pixelSize), applicationChannel: applicationChannel)
  NSGraphicsContext.restoreGraphicsState()

  guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
    throw IconRenderError.pngEncodingFailed(pixelSize)
  }
  return pngData
}

// Nebula palette, shared with site/assets/mark.svg so the Dock, the menu bar
// and the public site show one identity: a violet plate and silver marks.
private let nebulaViolet = NSColor(srgbRed: 0x6A / 255, green: 0x3F / 255, blue: 0xC4 / 255, alpha: 1)
private let printSilver = NSColor(srgbRed: 0xE4 / 255, green: 0xE0 / 255, blue: 0xD6 / 255, alpha: 1)

// The mark is specified on a 64-unit plate with a top-left origin, matching
// site/assets/mark.svg, so its coordinates can be checked against the SVG.
private struct MarkPlate {
  let tileRectangle: NSRect

  var unitLength: CGFloat { tileRectangle.width / 64 }

  func point(markX: CGFloat, markY: CGFloat) -> NSPoint {
    NSPoint(
      x: tileRectangle.minX + markX * unitLength,
      y: tileRectangle.maxY - markY * unitLength
    )
  }

  func circle(markX: CGFloat, markY: CGFloat, markRadius: CGFloat) -> NSBezierPath {
    let center = point(markX: markX, markY: markY)
    let radius = markRadius * unitLength
    return NSBezierPath(
      ovalIn: NSRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)
    )
  }
}

private func drawAstronomicalMark(canvasSize: CGFloat, applicationChannel: ApplicationChannel) {
  NSColor.clear.setFill()
  NSRect(x: 0, y: 0, width: canvasSize, height: canvasSize).fill()

  // The inset and soft shadow follow the macOS app-tile grid so the Dock
  // presents the plate at the same visual weight as system icons.
  let tileInset = canvasSize * 0.035
  let markPlate = MarkPlate(
    tileRectangle: NSRect(
      x: tileInset, y: tileInset,
      width: canvasSize - tileInset * 2, height: canvasSize - tileInset * 2))
  let tilePath = NSBezierPath(
    roundedRect: markPlate.tileRectangle, xRadius: canvasSize * 0.22, yRadius: canvasSize * 0.22)

  NSGraphicsContext.saveGraphicsState()
  let tileShadow = NSShadow()
  tileShadow.shadowColor = NSColor.black.withAlphaComponent(0.42)
  tileShadow.shadowBlurRadius = canvasSize * 0.045
  tileShadow.shadowOffset = NSSize(width: 0, height: -canvasSize * 0.018)
  tileShadow.set()
  nebulaViolet.setFill()
  tilePath.fill()
  NSGraphicsContext.restoreGraphicsState()

  NSGraphicsContext.saveGraphicsState()
  tilePath.addClip()
  drawRisingDisc(markPlate: markPlate, applicationChannel: applicationChannel)
  drawReticle(markPlate: markPlate, canvasSize: canvasSize)
  NSGraphicsContext.restoreGraphicsState()
}

// The disc is centred off the plate so only its limb shows: a body larger
// than the frame that holds it, which is the product's promise of models
// larger than memory.
private func drawRisingDisc(markPlate: MarkPlate, applicationChannel: ApplicationChannel) {
  let discPath = markPlate.circle(markX: 8, markY: 92, markRadius: 62)
  switch applicationChannel {
  case .stable:
    printSilver.setFill()
    discPath.fill()
  case .development:
    // Development borrows the product's "streaming" texture instead of a text
    // badge: hatching survives at 16 px where letters do not, and it keeps the
    // silhouette identical so the two channels read as one product.
    drawHatching(inside: discPath, markPlate: markPlate)
    printSilver.setStroke()
    discPath.lineWidth = max(1, markPlate.unitLength * 2.4)
    discPath.stroke()
  }
}

private func drawHatching(inside discPath: NSBezierPath, markPlate: MarkPlate) {
  NSGraphicsContext.saveGraphicsState()
  discPath.addClip()
  // A floor of three pixels per stripe period keeps stripes from merging into
  // a flat grey at the 16 and 32 px sizes.
  let stripePeriod = max(3, markPlate.unitLength * 5)
  let stripeWidth = stripePeriod * 0.45
  let tileRectangle = markPlate.tileRectangle
  let stripePath = NSBezierPath()
  stripePath.lineWidth = stripeWidth
  var stripeOffset = -tileRectangle.height
  while stripeOffset < tileRectangle.width {
    stripePath.move(to: NSPoint(x: tileRectangle.minX + stripeOffset, y: tileRectangle.minY))
    stripePath.line(
      to: NSPoint(
        x: tileRectangle.minX + stripeOffset + tileRectangle.height, y: tileRectangle.maxY))
    stripeOffset += stripePeriod
  }
  printSilver.setStroke()
  stripePath.stroke()
  NSGraphicsContext.restoreGraphicsState()
}

private func drawReticle(markPlate: MarkPlate, canvasSize: CGFloat) {
  // Below 64 px the reticle arms fall under one pixel, so the small cut drops
  // them and enlarges the reticle dot, matching site/assets/favicon.svg.
  guard canvasSize >= 64 else {
    printSilver.setFill()
    markPlate.circle(markX: 44, markY: 20, markRadius: 7).fill()
    return
  }
  let armSegments: [(CGFloat, CGFloat, CGFloat, CGFloat)] = [
    (44, 11, 44, 17), (44, 23, 44, 29), (35, 20, 41, 20), (47, 20, 53, 20),
  ]
  let armPath = NSBezierPath()
  armPath.lineWidth = markPlate.unitLength * 2.4
  armPath.lineCapStyle = .round
  for (startX, startY, endX, endY) in armSegments {
    armPath.move(to: markPlate.point(markX: startX, markY: startY))
    armPath.line(to: markPlate.point(markX: endX, markY: endY))
  }
  printSilver.setStroke()
  armPath.stroke()
  printSilver.setFill()
  markPlate.circle(markX: 44, markY: 20, markRadius: 2.8).fill()
}

private func renderIconFamily(arguments: IconRenderArguments) throws {
  try FileManager.default.createDirectory(
    at: arguments.outputDirectory,
    withIntermediateDirectories: true
  )
  for iconVariant in iconVariants {
    let pngData = try renderIcon(
      pixelSize: iconVariant.pixelSize,
      applicationChannel: arguments.applicationChannel
    )
    let outputFile = arguments.outputDirectory.appendingPathComponent(iconVariant.fileName)
    try pngData.write(to: outputFile, options: .atomic)
    print("[app-icon-renderer] file=\(iconVariant.fileName) pixels=\(iconVariant.pixelSize) status=success")
  }
}

do {
  let arguments = try parseArguments()
  try renderIconFamily(arguments: arguments)
} catch {
  let failureMessage: String
  if let localizedError = error as? LocalizedError,
    let localizedDescription = localizedError.errorDescription
  {
    failureMessage = localizedDescription
  } else {
    failureMessage = error.localizedDescription
  }
  FileHandle.standardError.write(Data("Error: \(failureMessage)\n".utf8))
  exit(1)
}
