//! The chat engine the worker holds: any chat family's engine behind one
//! boxed interface (issue #2473, Step 4). Every chat family prepares the same
//! request, and which engine serves it is decided once, when the factory
//! builds the engine and its processor together.

use crate::{DynInferenceEngine, TextInferenceRequest};

/// Whichever chat engine the factory built: Qwen 3.5 resident or streaming,
/// or a folded ternary pack.
pub type ModelFamilyInferenceEngine = Box<dyn DynInferenceEngine<TextInferenceRequest> + Sync>;

/// Boxes a chat family's engine as the engine the worker holds.
pub fn boxed_chat_engine<Engine>(chat_engine: Engine) -> ModelFamilyInferenceEngine
where
    Engine: DynInferenceEngine<TextInferenceRequest> + Sync + 'static,
{
    Box::new(chat_engine)
}
