#[cfg(feature = "serving-acceptance")]
#[path = "chat_serving/mod.rs"]
mod serving_acceptance;
#[cfg(feature = "serving-acceptance")]
#[path = "chat_serving/chat/small_dense_model.rs"]
mod small_dense_model;
#[cfg(feature = "serving-acceptance")]
mod support;
