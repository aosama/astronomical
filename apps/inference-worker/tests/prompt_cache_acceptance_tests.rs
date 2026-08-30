#[cfg(feature = "serving-acceptance")]
#[allow(dead_code)]
#[path = "serving_acceptance/chat/openai_rest.rs"]
mod openai_rest;
#[cfg(feature = "serving-acceptance")]
mod prompt_cache_acceptance;
#[cfg(feature = "serving-acceptance")]
#[allow(dead_code)]
#[path = "serving_acceptance/chat/small_dense_model.rs"]
mod small_dense_model;
#[cfg(feature = "serving-acceptance")]
mod support;
