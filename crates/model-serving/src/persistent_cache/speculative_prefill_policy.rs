/// Stored target, drafter, and keep-percentage identity used for targeted SSD
/// invalidation without deleting exact prompt state or unrelated model state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentSpeculativePrefillPolicyIdentity {
    target_model_id: String,
    target_model_revision: String,
    drafter_model_id: String,
    drafter_model_revision: String,
    keep_percentage: u32,
}

impl PersistentSpeculativePrefillPolicyIdentity {
    #[must_use]
    pub fn new(
        target_model_id: String,
        target_model_revision: String,
        drafter_model_id: String,
        drafter_model_revision: String,
        keep_percentage: u32,
    ) -> Self {
        Self {
            target_model_id,
            target_model_revision,
            drafter_model_id,
            drafter_model_revision,
            keep_percentage,
        }
    }

    #[must_use]
    pub fn target_model_id(&self) -> &str {
        &self.target_model_id
    }

    #[must_use]
    pub fn target_model_revision(&self) -> &str {
        &self.target_model_revision
    }

    #[must_use]
    pub fn drafter_model_id(&self) -> &str {
        &self.drafter_model_id
    }

    #[must_use]
    pub fn drafter_model_revision(&self) -> &str {
        &self.drafter_model_revision
    }

    #[must_use]
    pub const fn keep_percentage(&self) -> u32 {
        self.keep_percentage
    }

    #[must_use]
    pub fn should_purge_for_active_keep_percentage(
        &self,
        active_target_model_id: &str,
        active_target_model_revision: &str,
        active_drafter_model_id: &str,
        active_drafter_model_revision: &str,
        active_keep_percentage: u32,
    ) -> bool {
        self.target_model_id == active_target_model_id
            && self.target_model_revision == active_target_model_revision
            && self.drafter_model_id == active_drafter_model_id
            && self.drafter_model_revision == active_drafter_model_revision
            && self.keep_percentage != active_keep_percentage
    }
}
