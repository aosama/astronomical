import Foundation;
import AstronomicalConfig;

enum TemporaryDirectoryFixtureError: Error {
    case unrepresentableTemporaryDirectoryPath(String)
}

enum TestPaths {
    static func fromLiteral(_ pathLiteral: String) throws -> FilePath {
        guard !pathLiteral.isEmpty else {
            throw TemporaryDirectoryFixtureError.unrepresentableTemporaryDirectoryPath(pathLiteral);
        }
        return FilePath(string: pathLiteral);
    }

    static func canonicalized(_ path: FilePath) throws -> FilePath {
        let resolvedUrl: URL = URL(fileURLWithPath: path.string).resolvingSymlinksInPath();
        guard let resolvedPath: FilePath = FilePath(url: resolvedUrl) else {
            throw TemporaryDirectoryFixtureError.unrepresentableTemporaryDirectoryPath(path.string);
        }
        return resolvedPath;
    }
}

/** Creates and owns one unique temporary directory, removed on demand. */
final class TemporaryDirectoryFixture {
    private let rootDirectoryUrl: URL;
    private let rootDirectoryPathValue: FilePath;

    init() throws {
        let fixtureDirectoryName: String = "astronomical-instance-paths-\(UUID().uuidString)";
        let candidateRootUrl: URL = FileManager.default.temporaryDirectory.appending(path: fixtureDirectoryName);
        try FileManager.default.createDirectory(at: candidateRootUrl, withIntermediateDirectories: true);
        guard let candidateRootPath: FilePath = FilePath(url: candidateRootUrl) else {
            throw TemporaryDirectoryFixtureError.unrepresentableTemporaryDirectoryPath(candidateRootUrl.path);
        }
        self.rootDirectoryUrl = candidateRootUrl;
        self.rootDirectoryPathValue = candidateRootPath;
    }

    var rootDirectoryPath: FilePath {
        return self.rootDirectoryPathValue;
    }

    func destroy() throws {
        try FileManager.default.removeItem(at: self.rootDirectoryUrl);
    }
}
