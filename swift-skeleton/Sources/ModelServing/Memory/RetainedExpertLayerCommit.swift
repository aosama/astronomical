import Foundation

/// Commit decision plus the candidate page returned when the cache does not
/// take ownership of it.
public struct RetainedExpertLayerCommit<ExpertPage: ExpertWeightPage> {

    /// Decision reached for the offered page.
    public var outcome: RetainedExpertLayerCommitOutcome

    /// The offered page when ownership was refused (preserved or rejected);
    /// nil when the cache took ownership.
    public var uncommittedPage: ExpertPage?

    public init(
        outcome: RetainedExpertLayerCommitOutcome,
        uncommittedPage: ExpertPage?
    ) {
        self.outcome = outcome
        self.uncommittedPage = uncommittedPage
    }
}
