import XCTest

@testable import AstronomicalMenuCore

final class MaximumModelMemorySliderPresentationTests: XCTestCase {
  private let oneGigabyte: UInt64 = 1_073_741_824

  private func sliderPresentation(
    chosen: UInt64, minimum: UInt64 = 4, maximum: UInt64 = 36, modelSizeGigabytes: Double? = 28
  ) -> MaximumModelMemorySliderPresentation {
    MaximumModelMemorySliderPresentation(
      minimumWholeGigabytes: minimum, maximumWholeGigabytes: maximum,
      chosenWholeGigabytes: chosen,
      modelSizeBytes: modelSizeGigabytes.map { UInt64($0 * Double(oneGigabyte)) },
      macMemoryBytes: 48 * oneGigabyte)
  }

  func test_should_place_the_thumb_by_the_chosen_limit_across_the_range() {
    XCTAssertEqual(sliderPresentation(chosen: 4).chosenFraction, 0)
    XCTAssertEqual(sliderPresentation(chosen: 20).chosenFraction, 0.5, accuracy: 0.0001)
    XCTAssertEqual(sliderPresentation(chosen: 36).chosenFraction, 1)
  }

  func test_should_snap_a_point_on_the_track_to_a_whole_gigabyte_inside_the_range() {
    let presentation = sliderPresentation(chosen: 36)
    XCTAssertEqual(presentation.wholeGigabytes(atFraction: 0.5), 20)
    XCTAssertEqual(presentation.wholeGigabytes(atFraction: 0.51), 20)
    XCTAssertEqual(presentation.wholeGigabytes(atFraction: 0.52), 21)
    XCTAssertEqual(presentation.wholeGigabytes(atFraction: -3), 4)
    XCTAssertEqual(presentation.wholeGigabytes(atFraction: 7), 36)
  }

  func test_should_mark_the_model_size_only_when_it_lies_inside_the_range() {
    XCTAssertEqual(sliderPresentation(chosen: 36).modelMarkFraction ?? -1, 0.75, accuracy: 0.0001)
    XCTAssertEqual(sliderPresentation(chosen: 36).modelMarkCaption, "Model size 28.00 GB")
    XCTAssertNil(sliderPresentation(chosen: 36, modelSizeGigabytes: 40).modelMarkFraction)
    XCTAssertNil(sliderPresentation(chosen: 36, modelSizeGigabytes: 2).modelMarkFraction)
    XCTAssertNil(sliderPresentation(chosen: 36, modelSizeGigabytes: nil).modelMarkFraction)
    XCTAssertNil(sliderPresentation(chosen: 36, modelSizeGigabytes: nil).modelMarkCaption)
  }

  func test_should_say_what_the_limit_leaves_for_the_rest_of_the_mac() {
    let presentation = sliderPresentation(chosen: 30)
    XCTAssertFalse(presentation.partOfTheModelIsReadFromTheSSD)
    XCTAssertEqual(presentation.explanation, "Leaves 18 GB for macOS and your other apps.")
    XCTAssertEqual(
      sliderPresentation(chosen: 12, modelSizeGigabytes: nil).explanation,
      "Leaves 36 GB for macOS and your other apps.")
  }

  func test_should_say_that_the_model_streams_once_the_limit_is_below_its_size() {
    let presentation = sliderPresentation(chosen: 20)
    XCTAssertTrue(presentation.partOfTheModelIsReadFromTheSSD)
    XCTAssertEqual(
      presentation.explanation,
      "At 20 GB, part of this 28.00 GB model is read from the SSD, which is slower. Leaves 28 GB for other apps."
    )
    XCTAssertLessThanOrEqual(
      presentation.explanation.count, 110, "the sentence must fit the popover's two lines")
    // A limit equal to the size is not below it; a fraction of a gigabyte under it is.
    XCTAssertFalse(sliderPresentation(chosen: 28).partOfTheModelIsReadFromTheSSD)
    XCTAssertTrue(sliderPresentation(chosen: 28, modelSizeGigabytes: 28.05).partOfTheModelIsReadFromTheSSD)
  }

  func test_should_stay_defined_when_the_range_is_a_single_value_or_the_limit_passes_the_mac() {
    let singleValue = sliderPresentation(chosen: 8, minimum: 8, maximum: 8)
    XCTAssertEqual(singleValue.chosenFraction, 0)
    XCTAssertEqual(singleValue.wholeGigabytes(atFraction: 0.9), 8)
    XCTAssertNil(singleValue.modelMarkFraction)
    let pastTheMac = sliderPresentation(chosen: 64, maximum: 64, modelSizeGigabytes: nil)
    XCTAssertEqual(pastTheMac.explanation, "Leaves 0 GB for macOS and your other apps.")
  }

  private func commit(chosen: UInt64, applied: UInt64, configured: UInt64?) -> MaximumModelMemoryCommit {
    maximumModelMemoryCommit(
      chosenWholeGigabytes: chosen, maximumWholeGigabytes: 36, appliedWholeGigabytes: applied,
      configuredWholeGigabytes: configured)
  }

  func test_should_clear_the_setting_when_the_thumb_rests_at_the_top() {
    XCTAssertEqual(commit(chosen: 36, applied: 20, configured: 20), .followTheMacLimit)
    // A stored number equal to the top is cleared too, so the limit follows the Mac again.
    XCTAssertEqual(commit(chosen: 36, applied: 36, configured: 36), .followTheMacLimit)
    // A setting above what this Mac serves rests at the top; applying there clears it.
    XCTAssertEqual(commit(chosen: 36, applied: 36, configured: 64), .followTheMacLimit)
  }

  func test_should_save_the_number_for_any_position_below_the_top() {
    XCTAssertEqual(commit(chosen: 20, applied: 36, configured: nil), .setLimit(wholeGigabytes: 20))
    XCTAssertEqual(commit(chosen: 35, applied: 20, configured: 20), .setLimit(wholeGigabytes: 35))
    XCTAssertEqual(commit(chosen: 4, applied: 20, configured: 20), .setLimit(wholeGigabytes: 4))
  }

  func test_should_ask_for_nothing_when_letting_go_changes_nothing() {
    XCTAssertEqual(commit(chosen: 36, applied: 36, configured: nil), .nothing)
    XCTAssertEqual(commit(chosen: 20, applied: 20, configured: 20), .nothing)
  }
}
