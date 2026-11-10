import AppKit
import XCTest

@testable import AstronomicalMenuCore

final class AstronomicalBrandPaletteTests: XCTestCase {
  func test_should_resolve_the_icon_violet_under_a_light_appearance() throws {
    let lightAppearance = try XCTUnwrap(NSAppearance(named: .aqua))

    let resolvedAccent = AstronomicalBrandPalette.resolveAccent(for: lightAppearance)

    assertSRGBComponents(of: resolvedAccent, red: 106, green: 63, blue: 196)
  }

  func test_should_resolve_the_lifted_violet_under_a_dark_appearance() throws {
    let darkAppearance = try XCTUnwrap(NSAppearance(named: .darkAqua))

    let resolvedAccent = AstronomicalBrandPalette.resolveAccent(for: darkAppearance)

    assertSRGBComponents(of: resolvedAccent, red: 166, green: 140, blue: 255)
  }

  func test_should_treat_a_high_contrast_dark_appearance_as_dark() throws {
    let highContrastDarkAppearance = try XCTUnwrap(
      NSAppearance(named: .accessibilityHighContrastDarkAqua))

    let resolvedAccent = AstronomicalBrandPalette.resolveAccent(for: highContrastDarkAppearance)

    assertSRGBComponents(of: resolvedAccent, red: 166, green: 140, blue: 255)
  }

  func test_should_not_use_the_dark_violet_under_a_vibrant_light_appearance() throws {
    let vibrantLightAppearance = try XCTUnwrap(NSAppearance(named: .vibrantLight))

    let resolvedAccent = AstronomicalBrandPalette.resolveAccent(for: vibrantLightAppearance)

    XCTAssertNotEqual(resolvedAccent, AstronomicalBrandPalette.darkAppearanceAccent)
    assertSRGBComponents(of: resolvedAccent, red: 106, green: 63, blue: 196)
  }

  private func assertSRGBComponents(
    of color: NSColor, red: CGFloat, green: CGFloat, blue: CGFloat,
    file: StaticString = #filePath, line: UInt = #line
  ) {
    guard let sRGBColor = color.usingColorSpace(.sRGB) else {
      XCTFail("The accent has no sRGB form", file: file, line: line)
      return
    }
    XCTAssertEqual(sRGBColor.redComponent * 255, red, accuracy: 0.5, file: file, line: line)
    XCTAssertEqual(sRGBColor.greenComponent * 255, green, accuracy: 0.5, file: file, line: line)
    XCTAssertEqual(sRGBColor.blueComponent * 255, blue, accuracy: 0.5, file: file, line: line)
  }
}
