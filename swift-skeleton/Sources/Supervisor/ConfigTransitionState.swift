import Foundation;

import AstronomicalConfig;

/**
 * The supervisor's config-transition machinery, the Swift port of the
 * ApplicationState fields that coordinate live configuration mutations
 * (apps/supervisor/src/application.rs): the live reloadable snapshot, the
 * optional last-accepted configured snapshot, the transition lock that
 * serializes reload and memory mutations, and the newest memory generation
 * awaiting its terminal outcome.
 */
public final class ConfigTransitionState: @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private var reloadableConfigValue: ResolvedRuntimeConfig;
    private var configuredConfigSnapshotValue: ResolvedRuntimeConfig?;
    private var pendingMemoryConfigGenerationValue: String?;
    private var configurationValidationErrorValue: String?;
    private let configurationTransitionLock: NSLock = NSLock();

    public init(
        reloadableConfig: ResolvedRuntimeConfig,
        configuredConfigSnapshot: ResolvedRuntimeConfig?
    ) {
        self.reloadableConfigValue = reloadableConfig;
        self.configuredConfigSnapshotValue = configuredConfigSnapshot;
        self.pendingMemoryConfigGenerationValue = nil;
    }

    public func currentReloadableConfig() -> ResolvedRuntimeConfig {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.reloadableConfigValue;
    }

    public func replaceReloadableConfig(_ resolvedConfig: ResolvedRuntimeConfig) -> Void {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.reloadableConfigValue = resolvedConfig;
    }

    public func currentConfiguredConfigSnapshot() -> ResolvedRuntimeConfig? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.configuredConfigSnapshotValue;
    }

    public func replaceConfiguredConfigSnapshot(_ resolvedConfig: ResolvedRuntimeConfig) -> Void {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.configuredConfigSnapshotValue = resolvedConfig;
    }

    public func currentPendingMemoryConfigGeneration() -> String? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.pendingMemoryConfigGenerationValue;
    }

    public func setPendingMemoryConfigGeneration(_ configurationGeneration: String?) -> Void {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.pendingMemoryConfigGenerationValue = configurationGeneration;
    }

    /// The newest reload validation failure, surfaced by the status route
    /// until a later reload accepts the document.
    public func currentConfigurationValidationError() -> String? {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        return self.configurationValidationErrorValue;
    }

    public func setConfigurationValidationError(_ validationError: String?) -> Void {
        self.stateLock.lock();
        defer { self.stateLock.unlock(); }
        self.configurationValidationErrorValue = validationError;
    }

    /// Serializes one whole configuration transition: the guard spans the
    /// pending-generation check through the persisted commit and the worker
    /// update, mirroring configuration_transition_lock in application.rs.
    public func withTransitionGuard<TransitionOutcome>(
        _ transitionBody: () throws -> TransitionOutcome
    ) rethrows -> TransitionOutcome {
        self.configurationTransitionLock.lock();
        defer { self.configurationTransitionLock.unlock(); }
        return try transitionBody();
    }
}
