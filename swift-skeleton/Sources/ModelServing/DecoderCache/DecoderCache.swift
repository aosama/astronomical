// DecoderCache.swift — ModelServing.DecoderCache
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/decoder_cache/* —
// append_only_attention_state (+operations), rotating_attention_state
// (+rotating_layout), quantized_full_attention_state,
// gated_delta_recurrent_state, convolution_state, incremental_block_restore,
// layout, live_state, persistence_layouts, storage_geometry.
//
// Carried contracts:
// - Cache layouts and storage geometry derive from family config (layer
//   count, hidden size) — never from golden-master byte counts.
// - Block restore stays incremental so prefill resumption never rebuilds
//   the whole cache.
