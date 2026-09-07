use thiserror::Error;

/// Why a K2 Horizon MoVA `config.json` cannot become an executable family contract.
#[derive(Debug, Error)]
pub enum K2HorizonMoVAConfigError {
    #[error("failed to parse K2 Horizon MoVA config.json")]
    DeserializeConfig {
        #[source]
        source: serde_json::Error,
    },
    #[error("{description}")]
    InvalidConfigValue { description: String },
}
