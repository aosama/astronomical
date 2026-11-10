import XCTest

@testable import AstronomicalMenuCore

/// The agent reads the same feed the updater does; it must find the same
/// releases in it and must never read a broken download as "no update".
final class UpdateFeedTests: XCTestCase {
  private static let signedFeedText = """
    <?xml version="1.0" standalone="yes"?><!-- sparkle-sign-warning: signed -->
    <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" version="2.0">
      <channel>
        <title>Astronomical</title>
        <item>
          <title>0.3.0</title>
          <sparkle:version>2300</sparkle:version>
          <sparkle:shortVersionString>0.3.0</sparkle:shortVersionString>
          <sparkle:minimumSystemVersion>14.0</sparkle:minimumSystemVersion>
          <description sparkle:format="markdown"><![CDATA[## Notes with <angle> brackets]]></description>
          <enclosure url="https://example.invalid/Astronomical-0.3.0.dmg" length="1" type="application/octet-stream"></enclosure>
        </item>
        <item>
          <title>0.3.1 RC</title>
          <sparkle:version>2310</sparkle:version>
          <sparkle:shortVersionString>0.3.1</sparkle:shortVersionString>
          <sparkle:channel>release-candidate</sparkle:channel>
        </item>
      </channel>
    </rss><!-- sparkle-signatures:
    edSignature: x
    length: 1
    -->
    """

  func test_should_read_every_release_with_its_build_version_channel_and_minimum_system() throws {
    let feedItems = try UpdateFeedParser.items(fromFeedData: Data(Self.signedFeedText.utf8))

    XCTAssertEqual(
      feedItems,
      [
        UpdateFeedItem(
          buildNumber: 2300, displayVersion: "0.3.0", channelName: nil, minimumSystemVersion: "14.0"),
        UpdateFeedItem(
          buildNumber: 2310, displayVersion: "0.3.1", channelName: "release-candidate",
          minimumSystemVersion: nil),
      ])
  }

  func test_should_read_a_release_described_by_its_enclosure_attributes() throws {
    let enclosureFeedText = """
      <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel><item>
        <enclosure url="https://example.invalid/a.dmg" sparkle:version="2400" sparkle:shortVersionString="0.4.0"/>
      </item></channel></rss>
      """

    let feedItems = try UpdateFeedParser.items(fromFeedData: Data(enclosureFeedText.utf8))

    XCTAssertEqual(
      feedItems,
      [UpdateFeedItem(buildNumber: 2400, displayVersion: "0.4.0", channelName: nil, minimumSystemVersion: nil)])
  }

  func test_should_show_the_build_number_when_a_release_has_no_display_version() throws {
    let feedText = """
      <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel>
        <item><sparkle:version>7</sparkle:version></item>
      </channel></rss>
      """

    let feedItems = try UpdateFeedParser.items(fromFeedData: Data(feedText.utf8))

    XCTAssertEqual(feedItems.first?.displayVersion, "7")
  }

  func test_should_leave_out_a_release_whose_build_is_not_a_number() throws {
    let feedText = """
      <rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel>
        <item><sparkle:version>1.2-beta</sparkle:version></item>
        <item><title>no version at all</title></item>
        <item><sparkle:version>12</sparkle:version></item>
      </channel></rss>
      """

    let feedItems = try UpdateFeedParser.items(fromFeedData: Data(feedText.utf8))

    XCTAssertEqual(feedItems.map(\.buildNumber), [12])
  }

  func test_should_find_no_releases_in_a_feed_without_items() throws {
    let feedText = "<rss><channel><title>Astronomical</title></channel></rss>"

    XCTAssertEqual(try UpdateFeedParser.items(fromFeedData: Data(feedText.utf8)), [])
  }

  func test_should_refuse_a_download_that_is_not_a_feed() {
    let brokenDownloads = [
      "", "<html><body>Bad gateway", "<rss><channel><item></channel></rss>",
      // The updater's fields without the namespace that defines them.
      "<rss><channel><item><sparkle:version>7</sparkle:version></item></channel></rss>",
    ]

    for brokenDownload in brokenDownloads {
      XCTAssertThrowsError(
        try UpdateFeedParser.items(fromFeedData: Data(brokenDownload.utf8)), brokenDownload
      ) { thrownError in
        XCTAssertEqual(thrownError as? UpdateFeedError, .unreadableFeed)
      }
    }
  }
}
