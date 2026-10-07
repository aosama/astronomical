import Foundation;

/**
 * The opt-in gate for real-model journeys.
 *
 * Continues the Rust installed-model-contract convention
 * (`ASTRONOMICAL_QWEN_IMAGE_21_ARTIFACT_DIRECTORY`): a real-model journey
 * runs only when an environment variable names the installed artifact
 * directory on this machine, so no developer path is ever committed and
 * the default `swift test` run stays hermetic by construction.
 *
 * A real-model suite combines three traits:
 *
 *     @Suite(.serialized, .tags(.realModelJourney),
 *            .enabled(if: RealModelJourneyGate.qwen35ArtifactDirectory() != nil))
 *
 * The suite then resolves the directory once per journey through the same
 * accessor and fails closed if the directory no longer resolves.
 */
public enum RealModelJourneyGate {

    /// Resolves one installed-artifact directory from an environment
    /// variable; nil when the variable is unset or blank.
    public static func installedArtifactDirectory(
        environmentVariableName: String
    ) -> String? {
        return RealModelJourneyGate.nonBlankEnvironmentValue(
            environmentVariableName: environmentVariableName);
    }

    /// Resolves one non-blank value from an environment variable; nil when
    /// the variable is unset or blank. Artifact directories and plain
    /// opt-in flags share this resolution so no journey gate re-implements
    /// blank-value handling.
    private static func nonBlankEnvironmentValue(
        environmentVariableName: String
    ) -> String? {
        guard let rawValue: String = ProcessInfo.processInfo
            .environment[environmentVariableName]?
            .trimmingCharacters(in: .whitespacesAndNewlines) else {
            return nil;
        }
        if rawValue.isEmpty {
            return nil;
        }
        return rawValue;
    }

    /// The installed Qwen3.5 artifact directory for real-model journeys,
    /// resolved from `ASTRONOMICAL_QWEN35_ARTIFACT_DIRECTORY`.
    public static func qwen35ArtifactDirectory() -> String? {
        return RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: "ASTRONOMICAL_QWEN35_ARTIFACT_DIRECTORY");
    }

    /// The opt-in switch for the heavy SafeTensors concurrency journey,
    /// resolved from `ASTRONOMICAL_SAFETENSORS_CONCURRENCY_JOURNEY`. The
    /// journey writes and reads roughly 160 MiB (four 40 MiB tensors)
    /// without needing an installed model, so any non-blank value opts
    /// this machine in; it stays out of the default hermetic run by
    /// construction.
    public static func safetensorsConcurrencyJourneyEnabled() -> Bool {
        return RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: "ASTRONOMICAL_SAFETENSORS_CONCURRENCY_JOURNEY") != nil;
    }
}
