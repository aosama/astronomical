import Foundation;

import MLX;

/// Locates the mlx-swift SwiftPM shader bundle for test processes whose
/// main bundle is the swift-testing runner, where MLX's automatic bundle
/// lookup cannot see the resource. The search walks the package build tree
/// from the current working directory and never depends on an absolute or
/// machine-specific path.
public enum MLXMetallibLocator {

    private static let bundleName: String = "mlx-swift_Cmlx.bundle";
    private static let resourcePathComponents: Array<String> = [
        "Contents", "Resources", "default.metallib",
    ];
    private static let maximumSearchDepth: Int = 6;

    /// Points MLX at the discovered metallib; a no-op when the automatic
    /// lookup already resolves one or the bundle is not on disk.
    public static func overrideMetallibPathIfNecessary() -> Void {
        if GPU.metallib != nil {
            return;
        }
        guard let metallibUrl: URL = Self.locateDefaultMetallib() else {
            return;
        }
        GPU.metallib = metallibUrl;
    }

    /// Depth-limited search under the working directory for the SwiftPM
    /// resource bundle MLX ships its kernels in.
    public static func locateDefaultMetallib() -> URL? {
        let workingDirectoryUrl: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath);
        return Self.searchMetallib(under: workingDirectoryUrl, depth: 0);
    }

    private static func searchMetallib(
        under directoryUrl: URL,
        depth: Int
    ) -> URL? {
        if depth > MLXMetallibLocator.maximumSearchDepth {
            return nil;
        }
        let candidateMetallibUrl: URL = directoryUrl
            .appendingPathComponent(MLXMetallibLocator.bundleName)
            .appendingPathComponent(MLXMetallibLocator.resourcePathComponents.joined(separator: "/"));
        if FileManager.default.fileExists(atPath: candidateMetallibUrl.path) {
            return candidateMetallibUrl;
        }
        // The package build tree hides under the dotted `.build`, so hidden
        // entries are not skipped; the depth bound keeps the walk bounded.
        guard let childDirectoryUrls: [URL] = try? FileManager.default.contentsOfDirectory(
            at: directoryUrl, includingPropertiesForKeys: nil, options: []) else {
            return nil;
        }
        for childDirectoryUrl: URL in childDirectoryUrls {
            var isDirectory: ObjCBool = ObjCBool(false);
            FileManager.default.fileExists(
                atPath: childDirectoryUrl.path, isDirectory: &isDirectory);
            guard isDirectory.boolValue else {
                continue;
            }
            if let foundMetallibUrl: URL = Self.searchMetallib(
                under: childDirectoryUrl, depth: depth + 1) {
                return foundMetallibUrl;
            }
        }
        return nil;
    }
}
