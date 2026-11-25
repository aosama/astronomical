use crate::{AstroExpertCacheStatistics, InferenceEngineError};

use crate::MlxInferenceEngine;

use super::{Qwen3_5EngineState, Qwen3_5InferenceExecution, fatal_engine_error};

// These owner-thread operations intentionally live beside their asynchronous
// public wrappers. Keeping acceptance-only state mutation out of the engine
// owner prevents the production lifecycle module from becoming a test-control bag.
impl Qwen3_5EngineState {
    fn take_direct_expert_fetch_counts_for_tests(
        &self,
    ) -> Result<crate::DirectExpertFetchCounts, InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot count expert fetches before load"))?;
        Ok(loaded_model.direct_expert_fetching.tally.take())
    }

    fn session_expert_route_sample_for_tests(
        &self,
    ) -> Result<crate::SessionExpertRouteSample, InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot sample expert routes before load"))?;
        Ok(loaded_model.session_expert_route_sample())
    }

    fn take_routed_reply_recording_for_tests(
        &self,
    ) -> Result<Option<crate::RoutedReplyRecording>, InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot take a reply recording before load"))?;
        Ok(loaded_model.take_routed_reply_recording())
    }

    fn set_multi_token_prediction_draft_tokens_for_tests(
        &mut self,
        draft_tokens: u32,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_mut()
            .ok_or_else(|| fatal_engine_error("cannot set drafting before load"))?;
        loaded_model.base.chunking = loaded_model
            .base
            .chunking
            .with_multi_token_prediction_draft_tokens(draft_tokens);
        // The head's weights enter or leave the ledger's core weights.
        loaded_model.open_memory_governor();
        Ok(())
    }

    fn set_every_route_brought_to_the_host_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot choose route stops before load"))?;
        loaded_model
            .reply_measurement
            .brings_every_route_to_the_host
            .set(is_enabled);
        Ok(())
    }

    fn set_decode_misses_fetched_into_tables_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model =
            self.core.model.as_ref().ok_or_else(|| {
                fatal_engine_error("cannot choose the expert read path before load")
            })?;
        loaded_model
            .direct_expert_fetching
            .is_enabled
            .set(is_enabled);
        Ok(())
    }

    fn set_routes_looked_up_in_the_graph_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot choose the route lookup before load"))?;
        loaded_model.set_in_graph_table_serving(is_enabled);
        Ok(())
    }

    fn set_tables_grown_ahead_of_lookups_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot choose table growth before load"))?;
        loaded_model.set_tables_grown_ahead_of_lookups_for_tests(is_enabled);
        Ok(())
    }

    fn in_graph_serving_counts_for_tests(
        &self,
    ) -> Result<crate::InGraphServingCounts, InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot count in-graph lookups before load"))?;
        Ok(loaded_model.in_graph_table_serving().counts())
    }

    fn last_reply_stage_counts_for_tests(
        &self,
    ) -> Result<Option<crate::ReplyStageCounts>, InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("cannot read a reply's stages before load"))?;
        Ok(loaded_model.reply_measurement.stage_tally.last_reply())
    }

    fn astro_expert_cache_statistics_for_tests(
        &self,
    ) -> Result<AstroExpertCacheStatistics, InferenceEngineError> {
        self.core
            .model
            .as_ref()
            .map(|loaded_model| loaded_model.astro_expert_cache_statistics())
            .ok_or_else(|| fatal_engine_error("cannot inspect expert statistics before loading"))
    }

    fn complete_expert_payload_byte_count_for_tests(&self) -> Result<u64, InferenceEngineError> {
        let loaded_model =
            self.core.model.as_ref().ok_or_else(|| {
                fatal_engine_error("cannot inspect expert payload before loading")
            })?;
        let expert_pager = loaded_model.expert_pager.as_ref().ok_or_else(|| {
            fatal_engine_error("cannot inspect complete expert payload for a dense model")
        })?;
        expert_pager
            .complete_expert_payload_byte_count()
            .map_err(|expert_paging_error| {
                fatal_engine_error(format!(
                    "cannot measure complete expert payload: {expert_paging_error}"
                ))
            })
    }

    fn remove_resident_expert_source_files_for_tests(
        &mut self,
    ) -> Result<(), InferenceEngineError> {
        let loaded_model = self
            .core
            .model
            .as_mut()
            .ok_or_else(|| fatal_engine_error("cannot remove expert sources before loading"))?;
        let expert_pager = loaded_model.expert_pager.as_mut().ok_or_else(|| {
            fatal_engine_error("cannot remove resident expert sources from a dense model")
        })?;
        expert_pager.remove_resident_expert_source_files_for_tests();
        Ok(())
    }
}

/// The streaming engine's test controls: expert reads, route samples,
/// reply recordings and prompt-span redo. They are a trait because the
/// engine type is defined in the engine core.
// Every caller awaits these in place, so the futures' Send bound never matters.
#[allow(async_fn_in_trait)]
pub trait Qwen3_5StreamingEngineTestControls {
    /// Which way each fetching layer's missing experts went since this was
    /// last called, and why the direct way was refused when it was.
    async fn take_direct_expert_fetch_counts_for_tests(
        &self,
    ) -> Result<crate::DirectExpertFetchCounts, InferenceEngineError>;

    /// The session's always-on expert counters now; a reply's own figures
    /// are the change across it.
    async fn session_expert_route_sample_for_tests(
        &self,
    ) -> Result<crate::SessionExpertRouteSample, InferenceEngineError>;

    /// The routes of the last reply served with performance attribution on,
    /// and what the `astro_expert_cache` held when it began; `None` when no
    /// such reply was served since this was last called.
    async fn take_routed_reply_recording_for_tests(
        &self,
    ) -> Result<Option<crate::RoutedReplyRecording>, InferenceEngineError>;

    /// Sets how many tokens a model with a multi-token-prediction head
    /// drafts after each reply token; zero writes one token a forward.
    async fn set_multi_token_prediction_draft_tokens_for_tests(
        &self,
        draft_tokens: u32,
    ) -> Result<(), InferenceEngineError>;

    /// Makes every layer copy its route to the host, even a layer held
    /// whole, so a measurement can price the stop on its own.
    async fn set_every_route_brought_to_the_host_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError>;

    /// Chooses how a reply token's missing experts arrive: read straight into
    /// their tables (the default) or loaded as a page and copied afterwards.
    async fn set_decode_misses_fetched_into_tables_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError>;

    /// Chooses how a reply token finds its routed experts' table slots: by a
    /// lookup inside the token's graph (the default for a model that streams
    /// experts) or by copying each layer's route to the host.
    async fn set_routes_looked_up_in_the_graph_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError>;

    /// Stops or resumes growing tables between reply tokens. With it off,
    /// in-graph lookups refuse routes that need a larger table and those
    /// tokens are served again the exact way, which a contract checks.
    async fn set_tables_grown_ahead_of_lookups_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError>;

    /// Reply tokens attempted with in-graph lookups since the model loaded,
    /// how many were refused, and the layers the lookups served.
    async fn in_graph_serving_counts_for_tests(
        &self,
    ) -> Result<crate::InGraphServingCounts, InferenceEngineError>;

    /// Where the last finished reply's time went, by stage; `None` before
    /// any reply.
    async fn last_reply_stage_counts_for_tests(
        &self,
    ) -> Result<Option<crate::ReplyStageCounts>, InferenceEngineError>;

    /// Returns mode-neutral expert ownership plus cumulative paging statistics.
    async fn astro_expert_cache_statistics_for_tests(
        &self,
    ) -> Result<AstroExpertCacheStatistics, InferenceEngineError>;

    /// Returns the exact complete expert payload used by residency admission.
    async fn complete_expert_payload_byte_count_for_tests(
        &self,
    ) -> Result<u64, InferenceEngineError>;

    /// Removes resident-promotion source descriptors to prove typed failure recovery.
    async fn remove_resident_expert_source_files_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError>;

    /// Has the next prompt that is read one layer at a time fail the span at
    /// (`layer_index`, `span_index`) once after it is submitted, the way MLX
    /// reports a span it could not fit.
    async fn fail_prompt_span_once_for_tests(
        &self,
        layer_index: usize,
        span_index: usize,
    ) -> Result<(), InferenceEngineError>;

    /// Steps that hit the memory limit and redo attempts made since the
    /// model loaded, in that order.
    async fn memory_redo_counts_for_tests(&self) -> Result<(u64, u64), InferenceEngineError>;
}

impl Qwen3_5StreamingEngineTestControls for MlxInferenceEngine<Qwen3_5InferenceExecution> {
    async fn take_direct_expert_fetch_counts_for_tests(
        &self,
    ) -> Result<crate::DirectExpertFetchCounts, InferenceEngineError> {
        let (counts_sender, counts_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let counts = qwen_inference_execution.take_direct_expert_fetch_counts_for_tests();
            counts_sender
                .send(counts)
                .map_err(|_| fatal_engine_error("fetch counts receiver stopped unexpectedly"))
        })
        .await?;
        counts_receiver
            .recv()
            .map_err(|_| fatal_engine_error("fetch counts owner operation returned no response"))?
    }

    async fn session_expert_route_sample_for_tests(
        &self,
    ) -> Result<crate::SessionExpertRouteSample, InferenceEngineError> {
        let (sample_sender, sample_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let sample = qwen_inference_execution.session_expert_route_sample_for_tests();
            sample_sender.send(sample).map_err(|_| {
                fatal_engine_error("expert route sample receiver stopped unexpectedly")
            })
        })
        .await?;
        sample_receiver.recv().map_err(|_| {
            fatal_engine_error("expert route sample owner operation returned no response")
        })?
    }

    async fn take_routed_reply_recording_for_tests(
        &self,
    ) -> Result<Option<crate::RoutedReplyRecording>, InferenceEngineError> {
        let (recording_sender, recording_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let recording = qwen_inference_execution.take_routed_reply_recording_for_tests();
            recording_sender
                .send(recording)
                .map_err(|_| fatal_engine_error("reply recording receiver stopped unexpectedly"))
        })
        .await?;
        recording_receiver.recv().map_err(|_| {
            fatal_engine_error("reply recording owner operation returned no response")
        })?
    }

    async fn set_multi_token_prediction_draft_tokens_for_tests(
        &self,
        draft_tokens: u32,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.set_multi_token_prediction_draft_tokens_for_tests(draft_tokens)
        })
        .await
    }

    async fn set_every_route_brought_to_the_host_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.set_every_route_brought_to_the_host_for_tests(is_enabled)
        })
        .await
    }

    async fn set_decode_misses_fetched_into_tables_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.set_decode_misses_fetched_into_tables_for_tests(is_enabled)
        })
        .await
    }

    async fn set_routes_looked_up_in_the_graph_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.set_routes_looked_up_in_the_graph_for_tests(is_enabled)
        })
        .await
    }

    async fn set_tables_grown_ahead_of_lookups_for_tests(
        &self,
        is_enabled: bool,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.set_tables_grown_ahead_of_lookups_for_tests(is_enabled)
        })
        .await
    }

    async fn in_graph_serving_counts_for_tests(
        &self,
    ) -> Result<crate::InGraphServingCounts, InferenceEngineError> {
        let (counts_sender, counts_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let counts = qwen_inference_execution.in_graph_serving_counts_for_tests();
            counts_sender
                .send(counts)
                .map_err(|_| fatal_engine_error("lookup counts receiver stopped unexpectedly"))
        })
        .await?;
        counts_receiver
            .recv()
            .map_err(|_| fatal_engine_error("lookup counts owner operation returned no response"))?
    }

    async fn last_reply_stage_counts_for_tests(
        &self,
    ) -> Result<Option<crate::ReplyStageCounts>, InferenceEngineError> {
        let (counts_sender, counts_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let counts = qwen_inference_execution.last_reply_stage_counts_for_tests();
            counts_sender
                .send(counts)
                .map_err(|_| fatal_engine_error("stage counts receiver stopped unexpectedly"))
        })
        .await?;
        counts_receiver
            .recv()
            .map_err(|_| fatal_engine_error("stage counts owner operation returned no response"))?
    }

    async fn astro_expert_cache_statistics_for_tests(
        &self,
    ) -> Result<AstroExpertCacheStatistics, InferenceEngineError> {
        let (expert_statistics_sender, expert_statistics_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let expert_statistics =
                qwen_inference_execution.astro_expert_cache_statistics_for_tests()?;
            expert_statistics_sender
                .send(expert_statistics)
                .map_err(|_| fatal_engine_error("expert statistics receiver stopped unexpectedly"))
        })
        .await?;
        expert_statistics_receiver.recv().map_err(|_| {
            fatal_engine_error("expert statistics owner operation returned no response")
        })
    }

    async fn complete_expert_payload_byte_count_for_tests(
        &self,
    ) -> Result<u64, InferenceEngineError> {
        let (complete_expert_payload_sender, complete_expert_payload_receiver) =
            std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let complete_expert_payload_bytes =
                qwen_inference_execution.complete_expert_payload_byte_count_for_tests()?;
            complete_expert_payload_sender
                .send(complete_expert_payload_bytes)
                .map_err(|_| {
                    fatal_engine_error("complete expert payload receiver stopped unexpectedly")
                })
        })
        .await?;
        complete_expert_payload_receiver.recv().map_err(|_| {
            fatal_engine_error("complete expert payload owner operation returned no response")
        })
    }

    async fn remove_resident_expert_source_files_for_tests(
        &self,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(|qwen_inference_execution| {
            qwen_inference_execution.remove_resident_expert_source_files_for_tests()
        })
        .await
    }

    async fn fail_prompt_span_once_for_tests(
        &self,
        layer_index: usize,
        span_index: usize,
    ) -> Result<(), InferenceEngineError> {
        self.run_owner_test_operation(move |qwen_inference_execution| {
            qwen_inference_execution.prompt_span_to_fail_once_for_tests =
                Some((layer_index, span_index));
            Ok(())
        })
        .await
    }

    async fn memory_redo_counts_for_tests(&self) -> Result<(u64, u64), InferenceEngineError> {
        let (redo_counts_sender, redo_counts_receiver) = std::sync::mpsc::sync_channel(1);
        self.run_owner_test_operation(move |qwen_inference_execution| {
            let memory_redo_counts = qwen_inference_execution
                .core
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("no model is loaded to count redos for"))?
                .memory_redo_counts();
            redo_counts_sender
                .send(memory_redo_counts)
                .map_err(|_| fatal_engine_error("redo counts receiver stopped unexpectedly"))
        })
        .await?;
        redo_counts_receiver
            .recv()
            .map_err(|_| fatal_engine_error("redo counts owner operation returned no response"))
    }
}
