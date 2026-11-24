//! The ModernBERT trunk every family built on ModernBERT shares: it turns a
//! batch of padded token sequences into final hidden states and knows
//! nothing about what they are used for. Laya reads decisions from it and the
//! `modernbert` family crate pools embeddings from it, so it lives in the
//! decision shell rather than in either family (issue #2473, Step 7).

pub mod attention_mask;
pub mod configuration;
pub mod token_batch;
pub mod trunk;
pub mod weight_layout;
