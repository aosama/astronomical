import Foundation;
import Darwin;

/**
 * Path-walking helpers that mirror the Rust `Path` semantics the model
 * discovery port depends on: component extraction over the raw string,
 * symlink-following file-type checks via `stat`, and component-wise path
 * ordering (`Path: Ord`), which differs from plain string byte ordering for
 * paths containing "." components.
 */
internal enum DiscoveryPathNavigation {
    /**
     * Final path component, or nil for the filesystem root, mirroring
     * `Path::file_name`.
     */
    internal static func lastComponentName(of path: FilePath) -> String? {
        var trimmedPathString: String = path.string;
        while trimmedPathString.count > 1 && trimmedPathString.hasSuffix("/") {
            trimmedPathString.removeLast();
        }
        if trimmedPathString.isEmpty || trimmedPathString == "/" {
            return nil;
        }
        guard let lastSeparatorIndex: String.Index = trimmedPathString.lastIndex(of: "/") else {
            return trimmedPathString;
        }
        let componentIndex: String.Index = trimmedPathString.index(after: lastSeparatorIndex);
        let componentName: String = String(trimmedPathString[componentIndex...]);
        if componentName.isEmpty {
            return nil;
        }
        return componentName;
    }

    /**
     * The path itself followed by every ancestor up to the root, mirroring
     * `Path::ancestors`.
     */
    internal static func ancestorDirectoryPaths(startingFrom path: FilePath) -> Array<FilePath> {
        var ancestorPaths: Array<FilePath> = Array<FilePath>([path]);
        var currentPath: FilePath = path;
        while true {
            guard
                let parentPath: FilePath = currentPath.parentDirectory(),
                !parentPath.string.isEmpty,
                parentPath != currentPath
            else {
                return ancestorPaths;
            }
            ancestorPaths.append(parentPath);
            currentPath = parentPath;
        }
    }

    /** `Path::is_file`: symlink-following regular-file check; stat failure means false. */
    internal static func isExistingRegularFile(path: FilePath) -> Bool {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, 0) == 0 else {
            return false;
        }
        return (pathStatus.st_mode & S_IFMT) == S_IFREG;
    }

    /** `Path::is_dir`: symlink-following directory check; stat failure means false. */
    internal static func isExistingDirectory(path: FilePath) -> Bool {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, 0) == 0 else {
            return false;
        }
        return (pathStatus.st_mode & S_IFMT) == S_IFDIR;
    }

    /** Symlink-refusing regular-file check, mirroring `fs::symlink_metadata`. */
    internal static func isSymlinkTargetedRegularFile(path: FilePath) -> Bool {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, Darwin.AT_SYMLINK_NOFOLLOW) == 0 else {
            return false;
        }
        return (pathStatus.st_mode & S_IFMT) == S_IFREG;
    }

    /** `DirEntry::metadata` + `is_dir`: symlink entries are not directories without following them. */
    internal static func isSymlinkTargetedDirectory(path: FilePath) -> Bool {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, Darwin.AT_SYMLINK_NOFOLLOW) == 0 else {
            return false;
        }
        return (pathStatus.st_mode & S_IFMT) == S_IFDIR;
    }

    /**
     * `Path::try_exists`: an absent path and an unstatable path are distinct
     * outcomes, so permission problems surface as thrown errors instead of a
     * silent false.
     */
    internal static func filesystemEntryExists(path: FilePath) throws -> Bool {
        var pathStatus: stat = stat();
        if Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, 0) == 0 {
            return true;
        }
        if errno == ENOENT {
            return false;
        }
        throw NSError(
            domain: NSPOSIXErrorDomain,
            code: Int(errno),
            userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
        );
    }

    /** Modification time encoded as nanoseconds since the Unix epoch, via `lstat`. */
    internal static func modificationTimeNanoseconds(path: FilePath) -> UInt64? {
        var pathStatus: stat = stat();
        guard Darwin.fstatat(Darwin.AT_FDCWD, path.string, &pathStatus, Darwin.AT_SYMLINK_NOFOLLOW) == 0 else {
            return nil;
        }
        let secondsSinceEpoch: UInt64 = UInt64(max(0, pathStatus.st_mtimespec.tv_sec));
        let nanosecondRemainder: UInt64 = UInt64(max(0, pathStatus.st_mtimespec.tv_nsec));
        return secondsSinceEpoch * 1_000_000_000 + nanosecondRemainder;
    }

    /**
     * Strict ordering matching Rust `PathBuf: Ord`, which compares path
     * components (repeated separators collapse, and a component-wise prefix
     * sorts first) rather than raw path bytes.
     */
    internal static func rustPathOrdered(_ firstPath: FilePath, _ secondPath: FilePath) -> Bool {
        let firstComponents: Array<String> = firstPath.string.split(separator: "/").map(String.init);
        let secondComponents: Array<String> = secondPath.string.split(separator: "/").map(String.init);
        let firstIsAbsolute: Bool = firstPath.string.hasPrefix("/");
        let secondIsAbsolute: Bool = secondPath.string.hasPrefix("/");
        if firstIsAbsolute != secondIsAbsolute {
            return firstIsAbsolute;
        }
        let sharedComponentCount: Int = min(firstComponents.count, secondComponents.count);
        for componentOffset: Int in 0..<sharedComponentCount {
            let firstComponent: String = firstComponents[componentOffset];
            let secondComponent: String = secondComponents[componentOffset];
            if firstComponent.utf8.lexicographicallyPrecedes(secondComponent.utf8) {
                return true;
            }
            if secondComponent.utf8.lexicographicallyPrecedes(firstComponent.utf8) {
                return false;
            }
        }
        return firstComponents.count < secondComponents.count;
    }
}
