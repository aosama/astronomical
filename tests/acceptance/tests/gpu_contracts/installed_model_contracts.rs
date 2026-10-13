#[cfg(feature = "direct-mlx")]
mod common;
#[cfg(feature = "direct-mlx")]
#[path = "installed_model_contracts/mod.rs"]
mod installed_model_contracts;
#[cfg(feature = "direct-mlx")]
#[path = "../chat_serving/installed_journeys/mod.rs"]
mod serving_acceptance;
