import Foundation

/// Exact files and directory prefixes that belong to one executable model
/// package, migrating apps/supervisor/src/library/download_path_selection.rs:
/// the release catalog may narrow a repository tree to an explicit selection,
/// and overlapping or unsafe selectors are rejected at parse time.
public struct DownloadPathSelection: Equatable, Sendable {

    private let includedPaths: Array<String>?

    init(includedPaths: Array<String>?) throws {
        guard let includedPaths: Array<String> = includedPaths else {
            self.includedPaths = nil
            return
        }
        if includedPaths.isEmpty || includedPaths.count > DownloadPathSelection.MAXIMUM_INCLUDED_PATH_COUNT {
            throw DownloadPathSelectionError.invalid
        }
        var validatedPaths: Array<String> = Array<String>()
        for includedPath: String in includedPaths {
            guard DownloadPathSelection.isValidIncludedPath(includedPath) else {
                throw DownloadPathSelectionError.invalid
            }
            let normalizedPath: String = includedPath.lowercased()
            if validatedPaths.contains(where: { (existingPath: String) -> Bool in
                return DownloadPathSelection.selectorsOverlap(firstPath: existingPath, secondPath: normalizedPath)
            }) {
                throw DownloadPathSelectionError.invalid
            }
            validatedPaths.append(normalizedPath)
        }
        self.includedPaths = validatedPaths
    }

    /// Whether a validated Hub file belongs to the executable package.
    public func includes(relativePath: String) -> Bool {
        guard let includedPaths: Array<String> = self.includedPaths else {
            return true
        }
        let normalizedPath: String = relativePath.lowercased()
        return includedPaths.contains { (includedPath: String) -> Bool in
            if includedPath.hasSuffix("/") {
                return normalizedPath.hasPrefix(includedPath)
            }
            return normalizedPath == includedPath
        }
    }

    private static let MAXIMUM_INCLUDED_PATH_COUNT: Int = 64
    private static let MAXIMUM_INCLUDED_PATH_BYTES: Int = 1_024

    private static func isValidIncludedPath(_ includedPath: String) -> Bool {
        guard !includedPath.isEmpty,
            includedPath.utf8.count <= DownloadPathSelection.MAXIMUM_INCLUDED_PATH_BYTES,
            includedPath.allSatisfy({ (pathCharacter: Character) -> Bool in
                return pathCharacter.isASCII && !pathCharacter.isControlCharacter
            }),
            !includedPath.contains("\\"),
            !includedPath.hasPrefix("/") else {
            return false
        }
        let pathWithoutDirectoryMarker: String = includedPath.hasSuffix("/")
            ? String(includedPath.dropLast())
            : includedPath
        if pathWithoutDirectoryMarker.isEmpty || pathWithoutDirectoryMarker.hasSuffix("/") {
            return false
        }
        return DownloadPathSelection.isCanonicalAsciiPath(pathWithoutDirectoryMarker)
    }

    private static func isCanonicalAsciiPath(_ relativePath: String) -> Bool {
        let pathComponents: Array<Substring> = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        return !relativePath.isEmpty
            && relativePath.utf8.count <= DownloadPathSelection.MAXIMUM_INCLUDED_PATH_BYTES
            && relativePath.allSatisfy({ (pathCharacter: Character) -> Bool in
                return pathCharacter.isASCII && !pathCharacter.isControlCharacter
            })
            && !relativePath.contains("\\")
            && pathComponents.allSatisfy({ (pathComponent: Substring) -> Bool in
                return !pathComponent.isEmpty && pathComponent != "." && pathComponent != ".."
            })
    }

    private static func selectorsOverlap(firstPath: String, secondPath: String) -> Bool {
        let firstBase: String = firstPath.hasSuffix("/") ? String(firstPath.dropLast()) : firstPath
        let secondBase: String = secondPath.hasSuffix("/") ? String(secondPath.dropLast()) : secondPath
        if firstBase == secondBase {
            return true
        }
        if firstPath.hasSuffix("/") && secondPath.hasPrefix(firstPath) {
            return true
        }
        if secondPath.hasSuffix("/") && firstPath.hasPrefix(secondPath) {
            return true
        }
        return false
    }
}

enum DownloadPathSelectionError: Error {
    case invalid
}

extension Character {

    /// Control characters never appear in release-authored metadata or
    /// Hub paths; CharacterSet backs the check Unicode-correctly.
    var isControlCharacter: Bool {
        return self.unicodeScalars.contains { (unicodeScalar: Unicode.Scalar) -> Bool in
            return CharacterSet.controlCharacters.contains(unicodeScalar)
        }
    }
}
