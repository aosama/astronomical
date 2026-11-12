import Foundation
import XCTest

/// Reads production source text for the contracts that can only be stated
/// about the source itself (see `SourceWiringContractTests`).
///
/// Files are found by name anywhere under `Sources`, so moving a file between
/// folders never breaks a contract; only renaming or deleting it does, which
/// is exactly when the contract needs a second look.
enum PackageSources {
  /// `Tests/AstronomicalMenuContractTests/Support/` sits three folders below
  /// the package directory.
  static let packageDirectoryURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  /// The package lives at `apps/astronomical-menu` in the repository.
  static let repositoryRootURL = packageDirectoryURL
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  static func text(ofSourceFileNamed sourceFileName: String) throws -> String {
    let matchingSourceFileURLs = try swiftFileURLs(under: "Sources")
      .filter { $0.lastPathComponent == sourceFileName }
    XCTAssertEqual(
      matchingSourceFileURLs.count, 1, "\(sourceFileName) must exist exactly once under Sources")
    return try String(contentsOf: try XCTUnwrap(matchingSourceFileURLs.first), encoding: .utf8)
  }

  /// Every Swift file below a package-relative directory, however deeply nested.
  static func swiftFileURLs(under packageRelativeDirectory: String) throws -> [URL] {
    let directoryURL = packageDirectoryURL.appendingPathComponent(packageRelativeDirectory)
    let directoryEnumerator = try XCTUnwrap(
      FileManager.default.enumerator(at: directoryURL, includingPropertiesForKeys: nil),
      "\(packageRelativeDirectory) must exist in the package")
    var foundSwiftFileURLs: [URL] = []
    for case let fileURL as URL in directoryEnumerator where fileURL.pathExtension == "swift" {
      foundSwiftFileURLs.append(fileURL)
    }
    return foundSwiftFileURLs
  }
}
