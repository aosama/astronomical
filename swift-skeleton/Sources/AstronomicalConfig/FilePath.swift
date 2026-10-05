import Foundation;

/**
 * Immutable file-system path mirroring the Rust `PathBuf` semantics the
 * configuration unit depends on: plain string joining on append with no
 * implicit canonicalization, and absolute or root classification by the
 * leading slash alone. Foundation offers no path value type with these
 * semantics, so the configuration module owns this minimal one.
 */
public struct FilePath: Equatable, CustomStringConvertible, Sendable {
    private let pathString: String;

    public init(string: String) {
        self.pathString = string;
    }

    public init?(url: URL) {
        guard url.isFileURL else {
            return nil;
        }
        self.pathString = url.path;
    }

    public var string: String {
        return self.pathString;
    }

    public var isAbsolute: Bool {
        return self.pathString.hasPrefix("/");
    }

    public var isRoot: Bool {
        return self.pathString == "/";
    }

    public func appending(component: String) -> FilePath {
        let joinSeparator: String = self.pathString.hasSuffix("/") ? "" : "/";
        return FilePath(string: self.pathString + joinSeparator + component);
    }

    /**
     * Parent directory, or nil for the filesystem root, mirroring the Rust
     * `Path::parent()` semantics the atomic config write relies on.
     */
    public func parentDirectory() -> FilePath? {
        if (self.isRoot) {
            return nil;
        }
        let parentDirectoryUrl: URL = URL(fileURLWithPath: self.pathString).deletingLastPathComponent();
        return FilePath(url: parentDirectoryUrl);
    }

    public var description: String {
        return self.pathString;
    }
}
