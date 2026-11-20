import XCTest

@testable import AstronomicalMenuCore

final class PopoverFooterPresentationTests: XCTestCase {
  private let readyStatusDocument = DaemonStatusDocument(
    status: "ready", activity: "idle", readyModelIdentifier: nil, progress: nil,
    expertMemoryMode: nil, mlxMemoryCeilingBytes: 0, servingSession: .empty)

  func test_should_give_chat_the_only_accent_when_a_model_is_installed() {
    let footerPresentation = PopoverFooterPresentation(
      statusDocument: readyStatusDocument, hasDiscoveredModels: true)
    XCTAssertFalse(footerPresentation.offersModelDownload)
    XCTAssertTrue(footerPresentation.chatCarriesAccent)
    XCTAssertFalse(footerPresentation.offersRestartInPlaceOfObservatory)
  }

  func test_should_move_the_accent_to_the_download_button_when_no_model_is_installed() {
    let footerPresentation = PopoverFooterPresentation(
      statusDocument: readyStatusDocument, hasDiscoveredModels: false)
    XCTAssertTrue(footerPresentation.offersModelDownload)
    XCTAssertFalse(footerPresentation.chatCarriesAccent)
    XCTAssertFalse(footerPresentation.offersRestartInPlaceOfObservatory)
  }

  func test_should_offer_restart_and_no_download_while_the_engine_is_not_answering() {
    // A stopped engine reports no models, which must not read as "download one".
    let footerPresentation = PopoverFooterPresentation(
      statusDocument: .unavailable, hasDiscoveredModels: false)
    XCTAssertFalse(footerPresentation.offersModelDownload)
    XCTAssertFalse(footerPresentation.chatCarriesAccent)
    XCTAssertTrue(footerPresentation.offersRestartInPlaceOfObservatory)
  }

  func test_should_keep_observatory_while_the_engine_is_loading() {
    let loadingStatusDocument = DaemonStatusDocument(
      status: "loading", activity: "idle", readyModelIdentifier: nil, progress: nil,
      expertMemoryMode: nil, mlxMemoryCeilingBytes: 0, servingSession: .empty)
    let footerPresentation = PopoverFooterPresentation(
      statusDocument: loadingStatusDocument, hasDiscoveredModels: true)
    XCTAssertFalse(footerPresentation.offersRestartInPlaceOfObservatory)
    XCTAssertTrue(footerPresentation.chatCarriesAccent)
  }
}
