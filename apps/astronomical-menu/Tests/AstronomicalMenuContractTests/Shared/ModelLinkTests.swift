import Foundation
import XCTest

@testable import AstronomicalShared

/// A model link can be opened by any web page, so what it accepts is the
/// app's whole defence: one owner and one name, in Hugging Face's own
/// characters, for this channel only.
final class ModelLinkTests: XCTestCase {
  func test_should_round_trip_a_model_name_through_its_own_channel() throws {
    for applicationChannel in [ApplicationChannel.stable, .development, .appStore] {
      let linkURL = try XCTUnwrap(
        ModelLink.url(huggingfaceModelID: "example-org/Example.Model_4bit", channel: applicationChannel))

      XCTAssertEqual(
        ModelLink.huggingfaceModelID(from: linkURL, channel: applicationChannel),
        "example-org/Example.Model_4bit")
    }
  }

  func test_should_refuse_another_channels_link() throws {
    let stableURL = try XCTUnwrap(URL(string: "astronomical://models/huggingface/example-org/example"))

    XCTAssertNil(ModelLink.huggingfaceModelID(from: stableURL, channel: .development))
  }

  func test_should_refuse_a_link_without_exactly_one_owner_and_one_name() throws {
    let malformedLinks = [
      "astronomical://models/huggingface/example-org",
      "astronomical://models/huggingface/example-org/example/extra",
      "astronomical://models/elsewhere/example-org/example",
      "astronomical://chat/huggingface/example-org/example",
      "astronomical://models",
    ]
    for malformedLink in malformedLinks {
      let linkURL = try XCTUnwrap(URL(string: malformedLink))

      XCTAssertNil(ModelLink.huggingfaceModelID(from: linkURL, channel: .stable), malformedLink)
    }
  }

  func test_should_refuse_names_that_could_change_the_page_address() throws {
    let unsafeLinks = [
      "astronomical://models/huggingface/example-org/..",
      "astronomical://models/huggingface/../example",
      "astronomical://models/huggingface/example-org/name%3Fmodel%3Dother",
      "astronomical://models/huggingface/example-org/name%20with%20spaces",
      "astronomical://models/huggingface/example-org/name%26x%3D1",
    ]
    for unsafeLink in unsafeLinks {
      let linkURL = try XCTUnwrap(URL(string: unsafeLink))

      XCTAssertNil(ModelLink.huggingfaceModelID(from: linkURL, channel: .stable), unsafeLink)
    }
  }

  func test_should_refuse_a_name_longer_than_any_real_model_name() throws {
    let overlongName = String(repeating: "a", count: ModelLink.maximumModelIDCharacters)
    let linkURL = try XCTUnwrap(URL(string: "astronomical://models/huggingface/example-org/\(overlongName)"))

    XCTAssertNil(ModelLink.huggingfaceModelID(from: linkURL, channel: .stable))
  }

  func test_should_put_the_model_in_the_models_page_query() {
    XCTAssertEqual(
      ModelLink.observatoryPath(huggingfaceModelID: "example-org/Example.Model_4bit"),
      "/download-models?model=example-org/Example.Model_4bit")
  }
}
