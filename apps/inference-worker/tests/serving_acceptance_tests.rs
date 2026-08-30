#[cfg(feature = "serving-acceptance")]
mod serving_acceptance;
#[cfg(feature = "serving-acceptance")]
#[path = "serving_acceptance/chat/small_dense_model.rs"]
mod small_dense_model;
#[cfg(feature = "serving-acceptance")]
mod support;
