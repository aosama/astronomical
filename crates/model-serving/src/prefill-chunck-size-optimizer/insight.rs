/// Recent measured evidence for one requested prefill candidate in a context family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillChunckOptimizerCandidateEvidence {
    pub candidate_prefill_chunck_tokens: usize,
    pub observation_count: usize,
    pub average_actual_prefill_chunck_tokens: usize,
    pub average_elapsed_millis: u64,
    pub decisions_since_last_observation: Option<u64>,
}

/// Candidate evidence available to one context-aware optimizer decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunckOptimizerContextEvidence {
    pub has_observations_for_every_candidate: bool,
    pub candidate_evidence: Vec<PrefillChunckOptimizerCandidateEvidence>,
}
