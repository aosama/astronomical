#[cfg(feature = "serving-acceptance")]
#[path = "astro_ssd_prompt_cache/mod.rs"]
mod astro_ssd_prompt_cache_acceptance;
#[cfg(feature = "serving-acceptance")]
#[allow(dead_code, unused_imports)]
#[path = "chat_serving/chat/openai_rest.rs"]
mod openai_rest;
#[cfg(feature = "serving-acceptance")]
#[allow(dead_code)]
#[path = "chat_serving/chat/small_dense_model.rs"]
mod small_dense_model;
#[cfg(feature = "serving-acceptance")]
#[allow(dead_code)]
mod support;
