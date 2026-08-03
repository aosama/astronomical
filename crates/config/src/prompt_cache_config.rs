use std::path::PathBuf;

/// Resolved prompt-cache settings for worker startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptCacheConfig {
    global_prompt_cache_root_directory: PathBuf,
    active_model_prompt_cache_directory: PathBuf,
    global_prompt_cache_maximum_size_bytes: u64,
}

impl PromptCacheConfig {
    /// Builds the always-enabled prompt-cache policy.
    #[must_use]
    pub fn new(
        global_prompt_cache_root_directory: PathBuf,
        global_prompt_cache_maximum_size_bytes: u64,
    ) -> Self {
        Self {
            global_prompt_cache_root_directory: global_prompt_cache_root_directory.clone(),
            active_model_prompt_cache_directory: global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
        }
    }

    /// Returns the active model-and-revision directory when persistence is enabled.
    #[must_use]
    pub fn active_model_prompt_cache_directory(&self) -> &PathBuf {
        &self.active_model_prompt_cache_directory
    }

    /// Returns the one global prompt-cache root maximum in bytes.
    #[must_use]
    pub const fn global_prompt_cache_maximum_size_bytes(&self) -> u64 {
        self.global_prompt_cache_maximum_size_bytes
    }

    /// Returns the one global root shared by every model and revision.
    #[must_use]
    pub fn global_prompt_cache_root_directory(&self) -> &PathBuf {
        &self.global_prompt_cache_root_directory
    }

    /// Derives an isolated active directory while preserving the global root and maximum.
    #[must_use]
    pub fn for_model(&self, model_id: &str, revision: &str) -> PromptCacheConfig {
        PromptCacheConfig {
            global_prompt_cache_root_directory: self.global_prompt_cache_root_directory.clone(),
            active_model_prompt_cache_directory: self
                .global_prompt_cache_root_directory
                .join(model_id)
                .join(revision),
            global_prompt_cache_maximum_size_bytes: self.global_prompt_cache_maximum_size_bytes,
        }
    }
}
