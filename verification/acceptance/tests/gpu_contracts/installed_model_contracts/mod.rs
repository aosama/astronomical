#[cfg(feature = "direct-mlx")]
mod clef_decision_latency;
#[cfg(feature = "direct-mlx")]
mod clef_reference_decisions;
mod config;
#[cfg(feature = "direct-mlx")]
mod decision_memory_peak;
#[cfg(feature = "direct-mlx")]
mod decision_reference_fixture;
#[cfg(feature = "direct-mlx")]
mod dense_kept_layer_register_order;
#[cfg(feature = "direct-mlx")]
mod dense_layer_major_checkpoint_parity;
#[cfg(feature = "direct-mlx")]
mod dense_prompt_forward_memory;
#[cfg(feature = "direct-mlx")]
mod dense_reply_token_verdict;
#[cfg(feature = "direct-mlx")]
mod dense_streaming_speed_ladder;
#[cfg(feature = "direct-mlx")]
mod dflash_drafter_load_memory;
#[cfg(feature = "direct-mlx")]
mod expert_layer_at_a_time_checkpoint_parity;
#[cfg(feature = "direct-mlx")]
mod flash_next_greedy_parity;
#[cfg(feature = "direct-mlx")]
mod flash_next_loading;
#[cfg(feature = "direct-mlx")]
mod flash_next_sampled_forward_speed;
#[cfg(feature = "direct-mlx")]
mod greedy_answer;
#[cfg(feature = "direct-mlx")]
mod kev_decision_latency;
#[cfg(feature = "direct-mlx")]
mod kev_engine_requests;
#[cfg(feature = "direct-mlx")]
mod kev_reference_decisions;
#[cfg(feature = "direct-mlx")]
mod kokoro_reference_speech;
#[cfg(feature = "direct-mlx")]
mod kokoro_speech;
#[cfg(feature = "direct-mlx")]
mod laya_decision_latency;
#[cfg(feature = "direct-mlx")]
mod laya_engine_requests;
#[cfg(feature = "direct-mlx")]
mod laya_reference_decisions;
#[cfg(feature = "direct-mlx")]
mod modernbert_embedding_vectors;
#[cfg(feature = "direct-mlx")]
mod neural_engine_answer_tasks;
mod neural_engine_chunk_comparison;
mod neural_engine_mlp_prefill;
mod neural_engine_program_memory;
mod neural_engine_token_share_sweep;
mod ornith_load;
#[cfg(feature = "direct-mlx")]
mod phonon_speech;
#[cfg(feature = "direct-mlx")]
mod qwen_decision_readout;
mod qwen_image_21_conditioning;
#[cfg(feature = "direct-mlx")]
mod qwen_image_21_engine;
#[cfg(feature = "direct-mlx")]
mod qwen_image_21_render;
#[cfg(feature = "direct-mlx")]
mod qwen_image_21_text_encoder;
#[cfg(feature = "direct-mlx")]
mod qwen_image_21_transformer;
#[cfg(feature = "direct-mlx")]
mod qwen_image_21_vae;
#[cfg(feature = "direct-mlx")]
mod speech_audio_checks;
#[cfg(feature = "direct-mlx")]
mod streaming_fused_expert_combination;
mod tokenizer;
mod validate;
mod weights;
#[cfg(feature = "direct-mlx")]
mod whole_model_layer_recomputation;
#[cfg(feature = "direct-mlx")]
mod whole_model_pass_memory;
#[cfg(feature = "direct-mlx")]
mod whole_model_positive_control;
#[cfg(feature = "direct-mlx")]
mod whole_model_training;
mod whole_model_training_speed;
