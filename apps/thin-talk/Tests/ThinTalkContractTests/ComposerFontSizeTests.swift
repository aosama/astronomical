import Observation
import XCTest

@testable import ThinTalkCore

/// Contract for the composer font-zoom persistence layer: the documented
/// bounds, the fresh-install default, and that a deliberate size survives a
/// relaunch while out-of-range input is clamped instead of breaking the composer.
final class ComposerFontSizeTests: XCTestCase {
  func test_should_document_the_boundaries_and_default() {
    XCTAssertEqual(ComposerFontSize.defaultSize, 14)
    XCTAssertEqual(ComposerFontSize.minSize, 10)
    XCTAssertEqual(ComposerFontSize.maxSize, 24)
    XCTAssertEqual(ComposerFontSize.step, 1.0)
  }

  func test_should_start_a_fresh_install_on_the_default() {
    XCTAssertEqual(ComposerFontSize.load(from: isolatedDefaults()), ComposerFontSize.defaultSize)
  }

  func test_should_remember_a_deliberate_size_across_launches() {
    let defaults = isolatedDefaults()
    ComposerFontSize.save(20, to: defaults)
    XCTAssertEqual(ComposerFontSize.load(from: defaults), 20)
  }

  func test_should_clamp_out_of_range_stored_sizes_into_the_bounds() {
    let defaults = isolatedDefaults()
    ComposerFontSize.save(99, to: defaults)
    XCTAssertEqual(
      ComposerFontSize.load(from: defaults), ComposerFontSize.maxSize,
      "a stale or hand-edited oversized value must not render an unusable composer")
    ComposerFontSize.save(1, to: defaults)
    XCTAssertEqual(ComposerFontSize.load(from: defaults), ComposerFontSize.minSize)
  }

  private func isolatedDefaults() -> UserDefaults {
    let suite = "thin-talk.composer-font-size.tests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defaults.removePersistentDomain(forName: suite)
    return defaults
  }
}
