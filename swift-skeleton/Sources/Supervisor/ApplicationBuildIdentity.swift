import Foundation;

/**
 * Build provenance shown through every Astronomical control surface.
 *
 * Migrates apps/supervisor/src/application_build_identity.rs: the version,
 * build number, commit, and dirtiness the status document and the daemon
 * banner report. The Swift toolchain has no Cargo package metadata, so the
 * version falls back to the bundle's short version and then to 0.0.0-dev.
 */
public struct ApplicationBuildIdentity: Equatable, Sendable {

    /// The released version this build ships.
    public let version: String;
    /// Continuous-integration build number, zero when unbuilt locally.
    public let buildNumber: UInt64;
    /// The exact source revision, "unknown" when the build env omitted it.
    public let commit: String;
    /// Whether the working tree had uncommitted changes at build time.
    public let isDirty: Bool;

    public init(version: String, buildNumber: UInt64, commit: String, isDirty: Bool) {
        self.version = version;
        self.buildNumber = buildNumber;
        self.commit = commit;
        self.isDirty = isDirty;
    }

    /// Reads the build identity the toolchain environment stamped in.
    public static func current() -> ApplicationBuildIdentity {
        return ApplicationBuildIdentity.fromBuildEnvironment(
            buildEnvironment: ProcessInfo.processInfo.environment,
            bundleVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String);
    }

    /// Derives the identity from build-environment values, mirroring the
    /// option_env reads of the Rust implementation.
    public static func fromBuildEnvironment(
        buildEnvironment: Dictionary<String, String>,
        bundleVersion: String?
    ) -> ApplicationBuildIdentity {
        let buildNumber: UInt64 = buildEnvironment["ASTRONOMICAL_BUILD_NUMBER"].flatMap { (rawBuildNumber: String) -> UInt64? in
            return UInt64(rawBuildNumber);
        } ?? 0;
        return ApplicationBuildIdentity(
            version: bundleVersion ?? "0.0.0-dev",
            buildNumber: buildNumber,
            commit: buildEnvironment["ASTRONOMICAL_BUILD_COMMIT"] ?? "unknown",
            isDirty: buildEnvironment["ASTRONOMICAL_BUILD_DIRTY"] == "true");
    }
}
