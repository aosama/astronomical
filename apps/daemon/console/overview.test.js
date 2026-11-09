// Overview tests: the status band in the work's own units, the memory split,
// only the model facts that apply, the session tiles, and that nothing keeps
// describing a model after it unloads.

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");

const { createConsoleContext, attachRecordingElements } = require("./console-test-support.js");

function evaluate(scriptContext, expressionText, inputs = {}) {
    Object.assign(scriptContext, inputs);
    return JSON.parse(JSON.stringify(vm.runInContext(expressionText, scriptContext)));
}

test("should_show_the_output_ceiling_as_a_limit_while_generating", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "chat-model", progress: { phase: "generation", processed_tokens: 312, total_tokens: 65535, elapsed_ms: 7600 } }
    });

    assert.equal(nowPresentation.state, "working");
    assert.equal(nowPresentation.stateLabel, "Generating");
    assert.equal(nowPresentation.detail, "312 tokens written · up to 65,535 · 41 tok/s · 7.6 s");
    assert.equal(nowPresentation.fillPercent, 0, "a ceiling is not a target, so no bar");
});

test("should_count_prompt_tokens_read_and_clamp_the_bar_while_reading_the_prompt", () => {
    const scriptContext = createConsoleContext();
    scriptContext.neuralEnginePhaseSuffix = () => " · GPU";
    const nowPresentation = evaluate(scriptContext, "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "chat-model", progress: { phase: "target", processed_tokens: 12000, total_tokens: 10000, elapsed_ms: 2000 } }
    });

    assert.equal(nowPresentation.stateLabel, "Reading prompt · GPU");
    assert.equal(nowPresentation.detail, "12,000 of 10,000 prompt tokens read · 6000 tok/s · 2.0 s");
    assert.equal(nowPresentation.fillPercent, 100);
});

test("should_not_show_placeholder_tokens_while_preparing_to_write", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "chat-model", progress: { phase: "generation_preparation", processed_tokens: 0, total_tokens: 1, elapsed_ms: 1200 } }
    });

    assert.equal(nowPresentation.stateLabel, "Preparing");
    assert.equal(nowPresentation.detail, "Getting ready to write · 1.2 s");
});

test("should_treat_a_swap_as_loading_even_while_the_old_model_is_ready", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "ready", ready_model_id: "old-model", model_load_in_flight: { model_id: "new-model" },
            idle_model_unload: { after_seconds: 60, remaining_seconds: 30 } }
    });

    assert.deepEqual([nowPresentation.stateLabel, nowPresentation.modelTitle, nowPresentation.unloadText], ["Loading", "new-model", ""]);
});

test("should_say_the_worker_is_restarting_rather_than_idle_when_unavailable", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "unavailable", ready_model_id: null }
    });

    assert.equal(nowPresentation.state, "unavailable");
    assert.match(nowPresentation.detail, /restarts it/);
});

test("should_say_starting_not_loading_when_no_model_is_on_its_way", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "loading", ready_model_id: null }
    });

    assert.equal(nowPresentation.detail, "Starting the model worker.");
});

test("should_label_the_final_image_encode_as_saving", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "image-model", progress: { phase: "encoding_image", elapsed_ms: 100 } }
    });

    assert.equal(nowPresentation.stateLabel, "Saving image");
});

test("should_count_down_in_seconds_then_minutes_and_say_when_the_unload_is_due", () => {
    const scriptContext = createConsoleContext();
    const unloadTexts = evaluate(scriptContext, "[0, 119, 120, 3600, null].map((remaining) => overviewUnloadText({ after_seconds: 3600, remaining_seconds: remaining }))");

    assert.deepEqual(unloadTexts, ["Unloading now", "Unloads in 119 s", "Unloads in 2 min", "Unloads in 60 min", ""]);
});

test("should_count_steps_not_tokens_while_drawing_an_image", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "image-model", progress: { phase: "denoising", completed_steps: 7, total_steps: 28, elapsed_ms: 86051 } }
    });

    assert.equal(nowPresentation.stateLabel, "Drawing");
    assert.equal(nowPresentation.detail, "Step 7 of 28 · 86.1 s");
    assert.equal(nowPresentation.fillPercent, 25);
});

test("should_show_only_seconds_for_an_image_phase_without_steps", () => {
    const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
        statusForTest: { ready_model_id: "image-model", progress: { phase: "encoding_prompt", elapsed_ms: 1200 } }
    });

    assert.equal(nowPresentation.detail, "1.2 s");
    assert.equal(nowPresentation.fillPercent, 0);
});

test("should_never_show_an_internal_phase_name", () => {
    for (const unknownPhase of ["speculative_draft", "constructor"]) {
        const nowPresentation = evaluate(createConsoleContext(), "overviewNowPresentation(statusForTest)", {
            statusForTest: { ready_model_id: "chat-model", progress: { phase: unknownPhase, processed_tokens: 0, total_tokens: 1, elapsed_ms: 1 } }
        });

        assert.equal(nowPresentation.stateLabel, "Working", unknownPhase);
    }
});

test("should_say_when_an_idle_model_unloads_or_that_it_stays_loaded", () => {
    const scriptContext = createConsoleContext();
    const counting = evaluate(scriptContext, "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "ready", ready_model_id: "chat-model", idle_model_unload: { after_seconds: 60, remaining_seconds: 42 } }
    });
    const staying = evaluate(scriptContext, "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "ready", ready_model_id: "chat-model", idle_model_unload: { after_seconds: 0, remaining_seconds: null } }
    });

    assert.equal(counting.stateLabel, "Idle");
    assert.equal(counting.unloadText, "Unloads in 42 s");
    assert.equal(staying.unloadText, "Stays loaded");
});

test("should_show_loading_unavailable_and_no_model_without_an_unload_line", () => {
    const scriptContext = createConsoleContext();
    const loading = evaluate(scriptContext, "overviewNowPresentation(statusForTest)", {
        statusForTest: { status: "loading", ready_model_id: null, model_load_in_flight: { model_id: "next-model" } }
    });
    const unavailable = evaluate(scriptContext, "overviewNowPresentation(null)");
    const empty = evaluate(scriptContext, "overviewNowPresentation(statusForTest)", { statusForTest: { status: "ready", ready_model_id: null } });

    assert.deepEqual([loading.stateLabel, loading.modelTitle, loading.unloadText], ["Loading", "next-model", ""]);
    assert.equal(unavailable.state, "unavailable");
    assert.deepEqual([empty.modelTitle, empty.unloadText], ["No model loaded", ""]);
});

test("should_clamp_memory_segments_so_they_never_exceed_active_memory", () => {
    const segmentBytes = evaluate(createConsoleContext(), "reconciledMlxMemorySegmentBytes(snapshotForTest, 200)", {
        snapshotForTest: { active_memory_bytes: 100, expert_payload_bytes: 80, model_core_payload_bytes: 40, context_state_payload_bytes: 10 }
    });

    assert.deepEqual(segmentBytes, { activeMemoryBytes: 100, expertBytes: 80, modelCoreBytes: 20, contextStateBytes: 0, runtimeWorkBytes: 0, availableBytes: 100 });
});

test("should_attribute_unowned_active_memory_to_runtime_work", () => {
    const segmentBytes = evaluate(createConsoleContext(), "reconciledMlxMemorySegmentBytes(snapshotForTest, 200)", {
        snapshotForTest: { active_memory_bytes: 100, expert_payload_bytes: 20, model_core_payload_bytes: 30, context_state_payload_bytes: 10 }
    });

    assert.deepEqual(segmentBytes, { activeMemoryBytes: 100, expertBytes: 20, modelCoreBytes: 30, contextStateBytes: 10, runtimeWorkBytes: 40, availableBytes: 100 });
});

test("should_treat_a_missing_memory_snapshot_as_all_free", () => {
    const segmentBytes = evaluate(createConsoleContext(), "reconciledMlxMemorySegmentBytes(null, 200)");

    assert.deepEqual(segmentBytes, { activeMemoryBytes: 0, expertBytes: 0, modelCoreBytes: 0, contextStateBytes: 0, runtimeWorkBytes: 0, availableBytes: 200 });
});

test("should_map_every_macos_memory_pressure_state_and_unknown_input_safely", () => {
    const pressurePresentations = evaluate(createConsoleContext(),
        `["normal", "warning", "critical", null, "unexpected"].map(memoryPressurePresentationForState)`);

    assert.deepEqual(pressurePresentations.map((presentation) => presentation.state),
        ["normal", "warning", "critical", "unavailable", "unavailable"]);
});

test("should_list_chat_model_facts_without_any_dash_rows", () => {
    const modelFacts = evaluate(createConsoleContext(), "overviewModelFacts(statusForTest, listingForTest, placementForTest)", {
        statusForTest: { ready_model_id: "chat-model", ready_model_size_bytes: 23514046464 },
        listingForTest: { id: "chat-model", supported_endpoints: ["/v1/chat/completions"], context_window: 262144, input_modalities: ["text", "image"] },
        placementForTest: { stockSentence: "All in RAM", replySentence: "Experts are being served from RAM." }
    });

    assert.deepEqual(modelFacts, [
        { label: "Kind", value: "Chat" },
        { label: "Context", value: "262K tokens" },
        { label: "Reads", value: "Text and images" },
        { label: "On disk", value: "21.9 GB" },
        { label: "In memory", value: "All in RAM" },
        { label: "Experts", value: "Experts are being served from RAM." }
    ]);
});

test("should_leave_out_chat_only_facts_for_an_image_model", () => {
    const modelFacts = evaluate(createConsoleContext(), "overviewModelFacts(statusForTest, listingForTest, {})", {
        statusForTest: { ready_model_id: "image-model", ready_model_size_bytes: 10479720448 },
        listingForTest: { id: "image-model", supported_endpoints: ["/v1/images/generations"] }
    });

    assert.deepEqual(modelFacts.map((modelFact) => modelFact.label), ["Kind", "On disk"]);
    assert.equal(modelFacts[0].value, "Image generation");
});

test("should_list_no_facts_and_show_the_empty_note_once_nothing_is_loaded", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    elementsById["overview-model-facts"] = { hidden: false, replaceChildren(...factRows) { this.factRows = factRows; } };

    vm.runInContext("renderOverviewModel({ status: 'ready', ready_model_id: null }, {})", scriptContext);

    assert.deepEqual(elementsById["overview-model-facts"].factRows, []);
    assert.equal(elementsById["overview-model-facts"].hidden, true);
    assert.equal(elementsById["overview-model-empty"].hidden, false);
});

test("should_name_which_cache_each_reuse_figure_measures", () => {
    const scriptContext = createConsoleContext();
    const sessionPresentation = evaluate(scriptContext, "overviewSessionPresentation(sessionForTest)", {
        sessionForTest: { completed_request_count: 12, average_generation_tok_per_second: 67.4, total_prompt_token_count: 2490, total_reused_prompt_token_count: 2284 }
    });
    const promptCachePresentation = evaluate(scriptContext, "overviewPromptCachePresentation(settingForTest, cacheForTest)", {
        settingForTest: { enabled: { effective: true }, capacity_bytes: { effective: 53687091200 } },
        cacheForTest: { persistent_prompt_cache_hit_rate: 0.34, persistent_prompt_cache_hits: 34, persistent_prompt_cache_misses: 66, persistent_prompt_cache_total_size_bytes: 1288490188, persistent_prompt_cache_maximum_size_bytes: 0 }
    });

    assert.deepEqual(sessionPresentation, { requests: "12", speed: "67 tok/s", memoryReuse: "92%", memoryReuseHint: "2,284 of 2,490 prompt tokens" });
    assert.deepEqual(promptCachePresentation, { usage: "1.2 GB of 50.0 GB", hint: "34% of prompts found there" });
});

test("should_fall_back_to_the_configured_cache_setting_until_the_worker_acknowledges_it", () => {
    const scriptContext = createConsoleContext();
    const beforeAcknowledgement = evaluate(scriptContext,
        "overviewPromptCachePresentation({ enabled: { effective: null, configured: false, default: true } }, null)");
    const capacityFromDefault = evaluate(scriptContext,
        "overviewPromptCachePresentation({ enabled: { effective: null, default: true }, capacity_bytes: { effective: null, default: 53687091200 } }, null)");
    const beforeStatus = evaluate(scriptContext, "overviewPromptCachePresentation(null, { persistent_prompt_cache_total_size_bytes: 1 })");

    assert.equal(beforeAcknowledgement.usage, "Off");
    assert.equal(capacityFromDefault.usage, "0.0 GB of 50.0 GB");
    assert.equal(beforeStatus.usage, "—", "cache stats alone cannot say whether the cache is on");
});

test("should_switch_context_units_by_the_rounded_value", () => {
    const contextLabels = evaluate(createConsoleContext(), "[999, 1000, 999500, 1048576].map(formatOverviewContextTokens)");

    assert.deepEqual(contextLabels, ["999 tokens", "1K tokens", "1M tokens", "1M tokens"]);
});

test("should_draw_no_memory_segment_wider_than_the_bar_and_none_without_a_ceiling", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);

    vm.runInContext("renderOverviewMemory({ mlx_memory_ceiling_bytes: 0, mlx_memory_snapshot: { active_memory_bytes: 50, expert_payload_bytes: 50 } })", scriptContext);

    assert.equal(elementsById["overview-memory-seg-experts"].style.width, "0.0%");
    assert.equal(elementsById["overview-memory-free"].textContent, "0.0 GB");
});

test("should_render_one_term_and_value_per_model_fact", () => {
    const scriptContext = createConsoleContext();
    const elementsById = attachRecordingElements(scriptContext);
    elementsById["overview-model-facts"] = { hidden: true, replaceChildren(...factRows) { this.factRows = factRows; } };
    scriptContext.document.createElement = (tagName) => ({ tagName, textContent: "" });
    vm.runInContext(`rememberOverviewLoadedModelListing({ id: "image-model", supported_endpoints: ["/v1/images/generations"] });
        renderOverviewModel({ ready_model_id: "image-model", ready_model_size_bytes: 1073741824 }, {});`, scriptContext);

    assert.deepEqual(elementsById["overview-model-facts"].factRows.map((factRow) => [factRow.tagName, factRow.textContent]),
        [["dt", "Kind"], ["dd", "Image generation"], ["dt", "On disk"], ["dd", "1.0 GB"]]);
    assert.equal(elementsById["overview-model-empty"].hidden, true);
});

test("should_keep_only_the_size_when_the_model_list_has_not_arrived", () => {
    const modelFacts = evaluate(createConsoleContext(), "overviewModelFacts({ ready_model_id: 'chat-model', ready_model_size_bytes: 1073741824 }, null, {})");

    assert.deepEqual(modelFacts, [{ label: "On disk", value: "1.0 GB" }]);
});

test("should_clamp_reused_prompt_tokens_to_the_total", () => {
    const sessionPresentation = evaluate(createConsoleContext(), "overviewSessionPresentation({ total_prompt_token_count: 100, total_reused_prompt_token_count: 150 })");

    assert.equal(sessionPresentation.memoryReuse, "100%");
});

test("should_show_dashes_before_any_request_and_off_only_when_the_setting_is_off", () => {
    const scriptContext = createConsoleContext();
    const sessionPresentation = evaluate(scriptContext, "overviewSessionPresentation(undefined)");
    const switchedOff = evaluate(scriptContext, "overviewPromptCachePresentation({ enabled: { effective: false } }, null)");
    const onButUnused = evaluate(scriptContext,
        "overviewPromptCachePresentation({ enabled: { effective: true }, capacity_bytes: { effective: 53687091200 } }, { persistent_prompt_cache_maximum_size_bytes: 0 })");

    assert.deepEqual(sessionPresentation, { requests: "0", speed: "—", memoryReuse: "—", memoryReuseHint: "No prompts yet" });
    assert.equal(switchedOff.usage, "Off");
    assert.deepEqual(onButUnused, { usage: "0.0 GB of 50.0 GB", hint: "No prompts looked up yet" },
        "a zero maximum from the worker with no model loaded is not Off");
});
