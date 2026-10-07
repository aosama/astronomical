import Foundation

import AstronomicalConfig;

/// Display formatting shared by the CLI verbs, porting formatting.rs.
public enum CliFormatting {

    /// Decimal SI gigabytes (1 GB = 1,000,000,000 bytes): two decimals with
    /// trailing zeros and the trailing dot trimmed, so `1500000000` renders
    /// as `1.5` and `2000000000` as `2`.
    public static func formatGigabytes(_ sizeBytes: UInt64) -> String {
        let gigabytes: Double = Double(sizeBytes) / 1e9;
        var renderedText: String = String(format: "%.2f", gigabytes);
        while renderedText.hasSuffix("0") {
            renderedText.removeLast();
        }
        if renderedText.hasSuffix(".") {
            renderedText.removeLast();
        }
        return renderedText;
    }
}

/**
 * Resolves which loopback instance this CLI belongs to from the executable
 * path, porting instance.rs. `Astronomical Development.app` is checked first
 * because that path also contains the Stable bundle name as a substring.
 */
public enum CliInstanceResolution {

    /// Bundle folder names stamped by the macOS app assembler.
    private static let stableAppBundleName: String = "Astronomical.app";
    private static let developmentAppBundleName: String = "Astronomical Development.app";

    /// Infers runtime identity from where this binary lives. The input is
    /// canonicalized first because the installed CLI is reached through a
    /// symlink and the un-dereferenced link would classify every installed
    /// invocation as Development. A nonexistent path resolves to itself, the
    /// same fallback the Rust canonicalize uses.
    public static func runtimeInstanceFromExecutablePath(
        _ executablePath: String
    ) -> AstronomicalRuntimeInstance {
        let canonicalizedExecutablePath: String = (executablePath as NSString).resolvingSymlinksInPath;
        if canonicalizedExecutablePath.contains(CliInstanceResolution.developmentAppBundleName) {
            return .development;
        }
        if canonicalizedExecutablePath.contains(CliInstanceResolution.stableAppBundleName) {
            return .stable;
        }
        return .development;
    }

    /// Loopback instances to try for this binary, most preferred first. The
    /// other channel is kept as a fallback because an unpackaged developer
    /// build belongs to Development while the only running app may be Stable.
    public static func candidateInstances(
        _ preferredInstance: AstronomicalRuntimeInstance
    ) -> Array<AstronomicalRuntimeInstance> {
        switch (preferredInstance) {
        case .stable:
            return [.stable, .development];
        case .development:
            return [.development, .stable];
        }
    }
}
