#!/usr/bin/env swift

// Renders Astronomical's orbital identity with build-derived version and date
// badges into every PNG required by a macOS .iconset directory.

import AppKit
import Foundation

private struct IconRenderArguments {
  let outputDirectory: URL
  let applicationVersion: String
  let buildDate: String
}

private struct IconVariant {
  let fileName: String
  let pixelSize: Int
}

private enum IconRenderError: LocalizedError {
  case missingArgument(String)
  case unrecognizedArgument(String)
  case invalidVersion(String)
  case invalidBuildDate(String)
  case bitmapCreationFailed(Int)
  case graphicsContextCreationFailed(Int)
  case pngEncodingFailed(Int)

  var errorDescription: String? {
    switch self {
    case .missingArgument(let argumentName):
      return "missing required argument: \(argumentName)"
    case .unrecognizedArgument(let argument):
      return "unrecognized argument: \(argument)"
    case .invalidVersion(let version):
      return "version must be a semantic version, received: \(version)"
    case .invalidBuildDate(let buildDate):
      return "build date must be a real UTC calendar date in YYYYMMDD form, received: \(buildDate)"
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
  var applicationVersion: String?
  var buildDate: String?
  var argumentIndex = 0

  while argumentIndex < commandArguments.count {
    let argumentName = commandArguments[argumentIndex]
    guard argumentIndex + 1 < commandArguments.count else {
      throw IconRenderError.missingArgument(argumentName)
    }
    let argumentValue = commandArguments[argumentIndex + 1]
    switch argumentName {
    case "--output-directory":
      outputDirectoryPath = argumentValue
    case "--version":
      applicationVersion = argumentValue
    case "--build-date":
      buildDate = argumentValue
    default:
      throw IconRenderError.unrecognizedArgument(argumentName)
    }
    argumentIndex += 2
  }

  guard let outputDirectoryPath else {
    throw IconRenderError.missingArgument("--output-directory")
  }
  guard let applicationVersion else {
    throw IconRenderError.missingArgument("--version")
  }
  guard let buildDate else {
    throw IconRenderError.missingArgument("--build-date")
  }
  try validateApplicationVersion(applicationVersion)
  try validateBuildDate(buildDate)

  return IconRenderArguments(
    outputDirectory: URL(fileURLWithPath: outputDirectoryPath, isDirectory: true),
    applicationVersion: applicationVersion,
    buildDate: buildDate
  )
}

private func validateApplicationVersion(_ applicationVersion: String) throws {
  // Keep the rendered value aligned with Cargo semantic versions, including
  // optional pre-release and build metadata suffixes.
  let semanticVersionPattern = #"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"#
  let versionRange = NSRange(applicationVersion.startIndex..., in: applicationVersion)
  let semanticVersionExpression = try NSRegularExpression(pattern: semanticVersionPattern)
  guard semanticVersionExpression.firstMatch(in: applicationVersion, range: versionRange) != nil else {
    throw IconRenderError.invalidVersion(applicationVersion)
  }
}

private func validateBuildDate(_ buildDate: String) throws {
  guard buildDate.range(of: #"^[0-9]{8}$"#, options: .regularExpression) != nil else {
    throw IconRenderError.invalidBuildDate(buildDate)
  }
  let dateFormatter = DateFormatter()
  dateFormatter.calendar = Calendar(identifier: .gregorian)
  dateFormatter.locale = Locale(identifier: "en_US_POSIX")
  dateFormatter.timeZone = TimeZone(secondsFromGMT: 0)
  dateFormatter.dateFormat = "yyyyMMdd"
  dateFormatter.isLenient = false
  guard let parsedDate = dateFormatter.date(from: buildDate), dateFormatter.string(from: parsedDate) == buildDate else {
    throw IconRenderError.invalidBuildDate(buildDate)
  }
}

private func renderIcon(
  pixelSize: Int,
  applicationVersion: String,
  buildDate: String
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

  let canvasSize = CGFloat(pixelSize)
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = graphicsContext
  graphicsContext.imageInterpolation = .high
  NSColor.clear.setFill()
  NSRect(x: 0, y: 0, width: canvasSize, height: canvasSize).fill()

  // A generous inset lets macOS present the artwork as a native app tile
  // without clipping the orbital ring or its soft outer shadow.
  let tileInset = canvasSize * 0.035
  let tileRectangle = NSRect(
    x: tileInset,
    y: tileInset,
    width: canvasSize - (tileInset * 2),
    height: canvasSize - (tileInset * 2)
  )
  let tilePath = NSBezierPath(
    roundedRect: tileRectangle,
    xRadius: canvasSize * 0.22,
    yRadius: canvasSize * 0.22
  )
  let tileShadow = NSShadow()
  tileShadow.shadowColor = NSColor.black.withAlphaComponent(0.42)
  tileShadow.shadowBlurRadius = canvasSize * 0.045
  tileShadow.shadowOffset = NSSize(width: 0, height: -canvasSize * 0.018)
  tileShadow.set()
  NSColor(calibratedRed: 0.063, green: 0.071, blue: 0.059, alpha: 1).setFill()
  tilePath.fill()
  NSGraphicsContext.restoreGraphicsState()

  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = graphicsContext
  tilePath.addClip()

  // Sparse stars add depth while keeping the silhouette clean at small sizes.
  let starCoordinates: [(CGFloat, CGFloat, CGFloat, CGFloat)] = [
    (0.19, 0.76, 0.010, 0.55),
    (0.77, 0.72, 0.008, 0.44),
    (0.27, 0.48, 0.007, 0.34),
    (0.70, 0.42, 0.006, 0.30),
    (0.84, 0.55, 0.009, 0.48),
  ]
  for (horizontalPosition, verticalPosition, radiusScale, opacity) in starCoordinates {
    NSColor(calibratedWhite: 0.96, alpha: opacity).setFill()
    NSBezierPath(
      ovalIn: NSRect(
        x: canvasSize * horizontalPosition,
        y: canvasSize * verticalPosition,
        width: canvasSize * radiusScale * 2,
        height: canvasSize * radiusScale * 2
      )
    ).fill()
  }

  let orbitalCenter = NSPoint(x: canvasSize * 0.50, y: canvasSize * 0.62)
  let orbitalRectangle = NSRect(
    x: orbitalCenter.x - canvasSize * 0.34,
    y: orbitalCenter.y - canvasSize * 0.145,
    width: canvasSize * 0.68,
    height: canvasSize * 0.29
  )
  NSGraphicsContext.saveGraphicsState()
  let orbitalTransform = NSAffineTransform()
  orbitalTransform.translateX(by: orbitalCenter.x, yBy: orbitalCenter.y)
  orbitalTransform.rotate(byDegrees: -24)
  orbitalTransform.translateX(by: -orbitalCenter.x, yBy: -orbitalCenter.y)
  orbitalTransform.concat()
  let orbitalPath = NSBezierPath(ovalIn: orbitalRectangle)
  orbitalPath.lineWidth = max(1, canvasSize * 0.038)
  NSColor(calibratedRed: 0.957, green: 0.941, blue: 0.902, alpha: 0.96).setStroke()
  orbitalPath.stroke()
  NSGraphicsContext.restoreGraphicsState()

  let coreRadius = canvasSize * 0.115
  let coreRectangle = NSRect(
    x: orbitalCenter.x - coreRadius,
    y: orbitalCenter.y - coreRadius,
    width: coreRadius * 2,
    height: coreRadius * 2
  )
  let corePath = NSBezierPath(ovalIn: coreRectangle)
  NSGraphicsContext.saveGraphicsState()
  let coreShadow = NSShadow()
  coreShadow.shadowColor = NSColor(calibratedRed: 0.847, green: 1, blue: 0.243, alpha: 0.50)
  coreShadow.shadowBlurRadius = canvasSize * 0.075
  coreShadow.shadowOffset = .zero
  coreShadow.set()
  NSColor(calibratedRed: 0.847, green: 1, blue: 0.243, alpha: 1).setFill()
  corePath.fill()
  NSGraphicsContext.restoreGraphicsState()

  let satelliteRadius = canvasSize * 0.043
  let satelliteCenter = NSPoint(x: canvasSize * 0.79, y: canvasSize * 0.73)
  NSColor(calibratedRed: 1, green: 0.420, blue: 0.208, alpha: 1).setFill()
  NSBezierPath(
    ovalIn: NSRect(
      x: satelliteCenter.x - satelliteRadius,
      y: satelliteCenter.y - satelliteRadius,
      width: satelliteRadius * 2,
      height: satelliteRadius * 2
    )
  ).fill()

  drawIdentityBadge(
    canvasSize: canvasSize,
    applicationVersion: applicationVersion,
    buildDate: buildDate
  )
  NSGraphicsContext.restoreGraphicsState()

  guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
    throw IconRenderError.pngEncodingFailed(pixelSize)
  }
  return pngData
}

private func drawIdentityBadge(
  canvasSize: CGFloat,
  applicationVersion: String,
  buildDate: String
) {
  let badgeRectangle = NSRect(
    x: canvasSize * 0.14,
    y: canvasSize * 0.105,
    width: canvasSize * 0.72,
    height: canvasSize * 0.25
  )
  let badgePath = NSBezierPath(
    roundedRect: badgeRectangle,
    xRadius: canvasSize * 0.065,
    yRadius: canvasSize * 0.065
  )
  NSColor(calibratedRed: 0.957, green: 0.941, blue: 0.902, alpha: 0.97).setFill()
  badgePath.fill()

  // At 16 physical pixels, preserving the badge silhouette is clearer than
  // rasterizing illegible glyphs. Retina Finder variants use 64+ pixels and
  // therefore include both the exact version and the UTC build date.
  guard canvasSize >= 32 else { return }

  let centeredParagraph = NSMutableParagraphStyle()
  centeredParagraph.alignment = .center
  let versionFontSize = max(5, canvasSize * 0.105)
  let versionAttributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.systemFont(ofSize: versionFontSize, weight: .heavy),
    .foregroundColor: NSColor(calibratedRed: 0.063, green: 0.071, blue: 0.059, alpha: 1),
    .paragraphStyle: centeredParagraph,
  ]
  let versionRectangle = NSRect(
    x: badgeRectangle.minX,
    y: badgeRectangle.minY + badgeRectangle.height * 0.39,
    width: badgeRectangle.width,
    height: badgeRectangle.height * 0.50
  )
  (applicationVersion as NSString).draw(in: versionRectangle, withAttributes: versionAttributes)

  let dateFontSize = max(4, canvasSize * 0.052)
  let dateAttributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.monospacedDigitSystemFont(ofSize: dateFontSize, weight: .semibold),
    .foregroundColor: NSColor(calibratedRed: 0.78, green: 0.25, blue: 0.10, alpha: 1),
    .paragraphStyle: centeredParagraph,
    .kern: canvasSize * 0.004,
  ]
  let dateRectangle = NSRect(
    x: badgeRectangle.minX,
    y: badgeRectangle.minY + badgeRectangle.height * 0.08,
    width: badgeRectangle.width,
    height: badgeRectangle.height * 0.34
  )
  (buildDate as NSString).draw(in: dateRectangle, withAttributes: dateAttributes)
}

private func renderIconFamily(arguments: IconRenderArguments) throws {
  try FileManager.default.createDirectory(
    at: arguments.outputDirectory,
    withIntermediateDirectories: true
  )
  for iconVariant in iconVariants {
    let pngData = try renderIcon(
      pixelSize: iconVariant.pixelSize,
      applicationVersion: arguments.applicationVersion,
      buildDate: arguments.buildDate
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
  let failureMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
  FileHandle.standardError.write(Data("Error: \(failureMessage)\n".utf8))
  exit(1)
}
